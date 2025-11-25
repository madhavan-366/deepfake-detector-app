// main-backend/check_models.js
const axios = require('axios');
require('dotenv').config();

const API_KEY = process.env.GEMINI_API_KEY;
const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${API_KEY}`;

async function listModels() {
  try {
    console.log("Querying Google AI for available models...");
    const response = await axios.get(url);
    const models = response.data.models;
    
    console.log("\n✅ SUCCESS! Here are the models you can use:\n");
    
    models.forEach(model => {
      // The previous script had a bug here. This is the fixed line:
      if (model.supportedGenerationMethods && model.supportedGenerationMethods.includes('generateContent')) {
        console.log(`- ${model.name.replace('models/', '')}`);
      }
    });
  } catch (error) {
    console.error("\n❌ ERROR: Could not list models.");
    console.error(error.response ? error.response.data : error.message);
  }
}

listModels();