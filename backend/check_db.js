// backend/check_db.js
// Safe database connectivity diagnostic utility (never prints credentials)
const path = require('path');
const dns = require('dns').promises;
const mongoose = require('mongoose');

// Load .env from root directory
require('dotenv').config({ path: path.join(__dirname, '../.env') });

async function checkDatabaseConnection() {
  console.log('=== MongoDB Diagnostic Tool ===\n');

  const mongoUri = process.env.MONGO_URI;

  if (!mongoUri) {
    console.error('❌ Error: MONGO_URI is not defined in your .env file.');
    console.log('   Please set MONGO_URI in .env');
    process.exit(1);
  }

  // Parse connection URI safely (masking credentials)
  let hostname = '';
  let isSrv = false;
  try {
    const isSrvProtocol = mongoUri.startsWith('mongodb+srv://');
    isSrv = isSrvProtocol;
    const dummyUrl = new URL(mongoUri.replace('mongodb+srv://', 'http://').replace('mongodb://', 'http://'));
    hostname = dummyUrl.host;
    console.log(`📋 Protocol: ${isSrvProtocol ? 'mongodb+srv' : 'mongodb'}`);
    console.log(`📋 Hostname: ${hostname}`);
    console.log(`📋 Database: ${dummyUrl.pathname ? dummyUrl.pathname.replace('/', '') : '(default)'}`);
  } catch (err) {
    console.error('❌ Error: Could not parse MONGO_URI. Please verify its syntax.');
    process.exit(1);
  }

  // 1. DNS Resolution Test
  if (isSrv) {
    const srvTarget = `_mongodb._tcp.${hostname}`;
    console.log(`\n🔍 Step 1: Performing DNS SRV lookup for: ${srvTarget}`);
    try {
      const records = await dns.resolveSrv(srvTarget);
      console.log(`✅ DNS SRV records found (${records.length} servers):`, records.map(r => `${r.name}:${r.port}`).join(', '));
    } catch (dnsErr) {
      console.error(`❌ DNS SRV Lookup Failed: ${dnsErr.code} (${dnsErr.message})`);
      console.log('\n💡 Diagnosis:');
      console.log('   The domain "' + hostname + '" does not exist on public DNS.');
      console.log('   Common reasons in MongoDB Atlas:');
      console.log('   1. The cluster is PAUSED: Atlas free tier (M0) clusters pause after 60 days of inactivity.');
      console.log('      ➔ Solution: Go to cloud.mongodb.com and click "Resume".');
      console.log('   2. The cluster name or hostname was changed or mistyped.');
      console.log('      ➔ Solution: In Atlas, click "Connect" ➔ "Drivers" ➔ Copy the exact connection string.');
      console.log('   3. The cluster was deleted.');
      console.log('      ➔ Solution: Create a new free M0 cluster in Atlas and update your .env file.\n');
    }
  }

  // 2. Mongoose Connection Attempt
  console.log('🔄 Step 2: Testing Mongoose connection to configured URI (5s timeout)...');
  try {
    await mongoose.connect(mongoUri, { serverSelectionTimeoutMS: 5000 });
    console.log('✅ Connection SUCCESSFUL! MongoDB Atlas is connected and ready.\n');
    await mongoose.disconnect();
    process.exit(0);
  } catch (connErr) {
    console.error(`❌ Connection failed: ${connErr.message}`);
  }

  // 3. Check Local MongoDB availability as an immediate alternative
  console.log('\n🔍 Step 3: Checking if local MongoDB is available on 127.0.0.1:27017...');
  try {
    await mongoose.connect('mongodb://127.0.0.1:27017/deepfakeappdb', { serverSelectionTimeoutMS: 3000 });
    console.log('✅ Local MongoDB is running and responding on mongodb://127.0.0.1:27017/deepfakeappdb');
    console.log('   Tip: If your Atlas cluster is paused or unreachable, you can temporarily set:');
    console.log('   MONGO_URI=mongodb://127.0.0.1:27017/deepfakeappdb in your .env to continue working locally!');
    await mongoose.disconnect();
  } catch (localErr) {
    console.log('ℹ️ Local MongoDB is not responding:', localErr.message);
  }

  console.log('\n=== Diagnostic Completed ===');
}

checkDatabaseConnection();
