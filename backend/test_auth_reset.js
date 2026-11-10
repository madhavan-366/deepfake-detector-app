// backend/test_auth_reset.js
// Automated verification suite for Forgot Password / Reset Password implementation
const axios = require('axios');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });

const BASE_URL = 'http://localhost:3000';

async function runTests() {
  console.log('====================================================');
  console.log('   FORGOT / RESET PASSWORD COMPREHENSIVE TEST SUITE  ');
  console.log('====================================================\n');

  // Connect to local MongoDB fallback for user setup/check
  const mongoUri = 'mongodb://127.0.0.1:27017/deepfakeappdb';
  await mongoose.connect(mongoUri);
  const User = mongoose.model('User', new mongoose.Schema({ email: String, password: String }));

  // Pick two test accounts from the existing DB, or create them with known initial passwords if needed
  const testAccount1 = 'kmmadhavan37@gmail.com';
  const testAccount2 = 'operator_1793183804421@example.com';

  const user1 = await User.findOne({ email: testAccount1 });
  const user2 = await User.findOne({ email: testAccount2 });

  if (!user1 || !user2) {
    console.error('❌ Could not find test accounts in database:', { testAccount1: !!user1, testAccount2: !!user2 });
    process.exit(1);
  }

  // Set known initial passwords for both accounts so we can rigorously test before & after
  const initialPw1 = 'InitialPass123!';
  const initialPw2 = 'InitialPass456!';
  const salt = await bcrypt.genSalt(10);
  user1.password = await bcrypt.hash(initialPw1, salt);
  await user1.save();

  const salt2 = await bcrypt.genSalt(10);
  user2.password = await bcrypt.hash(initialPw2, salt2);
  await user2.save();

  console.log(`✅ Test Accounts Prepared:`);
  console.log(`   Account 1: ${testAccount1} (Initial password set)`);
  console.log(`   Account 2: ${testAccount2} (Initial password set)\n`);

  let passed = 0;
  let failed = 0;

  async function assertTest(name, fn) {
    try {
      await fn();
      console.log(`  ✅ [PASS] ${name}`);
      passed++;
    } catch (err) {
      console.error(`  ❌ [FAIL] ${name}:`, err.message || err);
      failed++;
    }
  }

  // 1. Initial login verification
  await assertTest('Account 1 logs in with initial password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/login`, {
      email: testAccount1,
      password: initialPw1,
    });
    if (!res.data.token) throw new Error('Token not received');
  });

  await assertTest('Account 2 logs in with initial password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/login`, {
      email: testAccount2,
      password: initialPw2,
    });
    if (!res.data.token) throw new Error('Token not received');
  });

  // 2. Validation error tests
  await assertTest('Reset fails on invalid email format', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/reset-password`, {
        email: 'not-an-email',
        newPassword: 'NewPassword123!',
        confirmPassword: 'NewPassword123!',
      });
      throw new Error('Should have failed');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
      if (!err.response?.data?.msg.includes('valid email')) throw new Error(`Unexpected message: ${err.response?.data?.msg}`);
    }
  });

  await assertTest('Reset fails on password shorter than 6 characters', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/reset-password`, {
        email: testAccount1,
        newPassword: '123',
        confirmPassword: '123',
      });
      throw new Error('Should have failed');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
      if (!err.response?.data?.msg.includes('at least 6 characters')) throw new Error(`Unexpected message: ${err.response?.data?.msg}`);
    }
  });

  await assertTest('Reset fails when passwords do not match', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/reset-password`, {
        email: testAccount1,
        newPassword: 'NewPassword123!',
        confirmPassword: 'DifferentPassword456!',
      });
      throw new Error('Should have failed');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
      if (!err.response?.data?.msg.includes('do not match')) throw new Error(`Unexpected message: ${err.response?.data?.msg}`);
    }
  });

  await assertTest('Reset fails when email is not registered in system (404)', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/reset-password`, {
        email: 'unregistered_ghost_user_99999@example.com',
        newPassword: 'NewPassword123!',
        confirmPassword: 'NewPassword123!',
      });
      throw new Error('Should have failed');
    } catch (err) {
      if (err.response?.status !== 404) throw new Error(`Expected 404 but got ${err.response?.status}`);
      if (!err.response?.data?.msg.includes('No registered account')) throw new Error(`Unexpected message: ${err.response?.data?.msg}`);
    }
  });

  await assertTest('Reset fails when new password matches existing password', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/reset-password`, {
        email: testAccount1,
        newPassword: initialPw1,
        confirmPassword: initialPw1,
      });
      throw new Error('Should have failed');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
      if (!err.response?.data?.msg.includes('cannot be the same')) throw new Error(`Unexpected message: ${err.response?.data?.msg}`);
    }
  });

  // 3. Reset Password on Account 1
  const updatedPw1 = 'SecureUpdatedPw1_2026!';
  await assertTest('Account 1: Successfully resets password via /api/auth/reset-password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/reset-password`, {
      email: testAccount1,
      newPassword: updatedPw1,
      confirmPassword: updatedPw1,
    });
    if (res.status !== 200 || !res.data.msg.includes('securely reset')) {
      throw new Error(`Unexpected response: ${JSON.stringify(res.data)}`);
    }
  });

  await assertTest('Account 1: Old password is now rejected', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/login`, {
        email: testAccount1,
        password: initialPw1,
      });
      throw new Error('Old password still accepted!');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
    }
  });

  await assertTest('Account 1: Immediate login works with new password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/login`, {
      email: testAccount1,
      password: updatedPw1,
    });
    if (!res.data.token) throw new Error('No JWT token returned');
  });

  // 4. Reset Password on Account 2
  const updatedPw2 = 'SecureUpdatedPw2_2026!';
  await assertTest('Account 2: Successfully resets password via /api/auth/reset-password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/reset-password`, {
      email: testAccount2,
      newPassword: updatedPw2,
      confirmPassword: updatedPw2,
    });
    if (res.status !== 200 || !res.data.msg.includes('securely reset')) {
      throw new Error(`Unexpected response: ${JSON.stringify(res.data)}`);
    }
  });

  await assertTest('Account 2: Old password is now rejected', async () => {
    try {
      await axios.post(`${BASE_URL}/api/auth/login`, {
        email: testAccount2,
        password: initialPw2,
      });
      throw new Error('Old password still accepted!');
    } catch (err) {
      if (err.response?.status !== 400) throw new Error(`Expected 400 but got ${err.response?.status}`);
    }
  });

  await assertTest('Account 2: Immediate login works with new password', async () => {
    const res = await axios.post(`${BASE_URL}/api/auth/login`, {
      email: testAccount2,
      password: updatedPw2,
    });
    if (!res.data.token) throw new Error('No JWT token returned');
  });

  // 5. Rate Limiting Test (Account 2 spam test)
  await assertTest('Rate limiting triggers HTTP 429 when abuse threshold exceeded', async () => {
    let got429 = false;
    for (let i = 0; i < 8; i++) {
      try {
        await axios.post(`${BASE_URL}/api/auth/reset-password`, {
          email: testAccount2,
          newPassword: `RapidAttempt${i}!123`,
          confirmPassword: `RapidAttempt${i}!123`,
        });
      } catch (err) {
        if (err.response?.status === 429) {
          got429 = true;
          break;
        }
      }
    }
    if (!got429) throw new Error('Rate limit 429 was not returned');
  });

  console.log('\n====================================================');
  console.log(`TEST SUMMARY: ${passed} Passed, ${failed} Failed`);
  console.log('====================================================\n');

  await mongoose.disconnect();
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
