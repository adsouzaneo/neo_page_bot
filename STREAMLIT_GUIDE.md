# 🚀 Quick Start Guide

## Your Streamlit Chatbot is Running!

### Access the App:
**Local URL:** http://localhost:8501

The app is now live in your browser!

## Features:

### ✨ Interactive Chat Interface
- Ask questions in natural language
- Get answers based ONLY on the 5 scraped NeoSapients pages
- See source citations for every answer
- Chat history preserved during session

### 📊 Built-in Features:
- **Example Questions** - Click sidebar buttons to try pre-made queries
- **Source Retrieval Control** - Adjust how many sources to retrieve (1-10)
- **Knowledge Base Info** - See what pages are included
- **Clear History** - Reset the conversation anytime

### 🎨 UI Elements:
- Clean, professional chat interface
- User messages in blue
- AI responses in gray
- Source citations highlighted in orange
- Relevance scores for transparency

## How to Use:

1. **Ask Questions** - Type in the input box at the bottom
2. **Click Send** or press Enter
3. **View Response** - AI answer with source citations
4. **Check Sources** - See which pages were used (with relevance %)

## Example Questions:
- "What is NeoSapients?"
- "What is Context OS?"
- "Tell me about AI agents"
- "What industries do you serve?"
- "What career opportunities are available?"

## Current Status:

✅ **Working (Without OpenAI):**
- Retrieves relevant content from Milvus
- Shows retrieved text snippets
- Displays source citations
- Full semantic search working

⚠️ **To Enable AI Answers:**
1. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
2. Restart the app:
   ```bash
   # Stop: Press Ctrl+C in terminal
   # Start: 
   /Users/alricrommeldsouza/Desktop/Rag_imp/.venv/bin/streamlit run app.py
   ```

## Commands:

**Start App:**
```bash
/Users/alricrommeldsouza/Desktop/Rag_imp/.venv/bin/streamlit run app.py
```

**Stop App:**
Press `Ctrl+C` in the terminal

**Restart App:**
1. Stop with `Ctrl+C`
2. Run start command again

## What Happens Without OpenAI Key:
- Shows retrieved context directly
- Still perfectly functional for exploring your knowledge base
- All semantic search features work
- Just no AI-generated summaries

## Performance Tips:
- First query takes ~2 seconds (loading model)
- Subsequent queries are fast (<1 second)
- Adjust "Number of sources" for speed vs accuracy

## Troubleshooting:

**Port already in use?**
```bash
# Kill existing streamlit process
pkill -f streamlit
# Then restart
```

**Can't access in browser?**
- Check the terminal for the correct URL
- Try http://localhost:8501

**Want to share?**
- Use the Network URL shown in terminal
- Others on same network can access

---

**Enjoy your RAG chatbot! 🎉**
