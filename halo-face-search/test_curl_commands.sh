#!/bin/bash

# Test Halo Face Search API on Railway
# Replace YOUR_RAILWAY_URL with your actual Railway URL

RAILWAY_URL="https://your-project-name.railway.app"

echo "🚀 Testing Halo Face Search API on Railway"
echo "🌐 URL: $RAILWAY_URL"
echo "================================================"

echo ""
echo "🏥 1. Health Check:"
curl -s "$RAILWAY_URL/health" | jq '.'

echo ""
echo "🏠 2. Root Endpoint:"
curl -s "$RAILWAY_URL/" | jq '.message, .status, .database_faces'

echo ""
echo "📊 3. Database Stats:"
curl -s "$RAILWAY_URL/stats" | jq '.database_stats'

echo ""
echo "📖 4. API Documentation (opens in browser):"
echo "   Visit: $RAILWAY_URL/docs"

echo ""
echo "🔍 5. Face Search Test:"
echo "   Upload an image to test face search:"
echo "   curl -X POST $RAILWAY_URL/search \\"
echo "     -F \"file=@your_image.jpg\" \\"
echo "     -F \"top_k=5\""

echo ""
echo "✅ Basic tests complete!"
echo "   For full testing, update RAILWAY_URL and run: python test_railway_api.py" 