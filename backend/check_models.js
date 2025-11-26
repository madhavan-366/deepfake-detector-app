const axios = require('axios');
const path = require('path');
// Look for .env in the MAIN folder (one level up)
require('dotenv').config({ path: path.join(__dirname, '../.env') });

const API_KEY = process.env.GEMINI_API_KEY;

if (!API_KEY) {
    console.error("❌ Error: Could not find GEMINI_API_KEY in the .env file.");
    process.exit(1);
}

async function checkModels() {
    console.log("🔍 Connecting to Google Brain to list models...");
    try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${API_KEY}`;
        const response = await axios.get(url);
        
        console.log("\n✅ AVAILABLE MODELS:");
        const models = response.data.models;
        
        // Filter only the "generateContent" models (Chat models)
        const chatModels = models.filter(m => m.supportedGenerationMethods.includes("generateContent"));
        
        chatModels.forEach(model => {
            // Clean up the name (remove "models/" prefix)
            const name = model.name.replace("models/", "");
            console.log(`- ${name}`);
        });

    } catch (error) {
        console.error("\n❌ Failed to list models.");
        if (error.response) {
            console.error(`Error ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`);
        } else {
            console.error(error.message);
        }
    }
}

checkModels();