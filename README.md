# Excel Chat with Gemini 🤖

A Streamlit application that allows you to chat with your Excel files using Google's Gemini Pro model. Upload any Excel file and ask questions about your data in natural language.

## Features

- 📊 Excel file upload and preview
- 💬 Natural language queries about your data
- 🤖 Powered by Google's Gemini Pro
- 🔍 Smart data context understanding
- 🚀 Real-time streaming responses
- 📝 Chat history persistence

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rais-th/excel-chat-gemini.git
cd excel-chat-gemini
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google API key:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace the API key in `app.py` or set it as an environment variable:
     ```bash
     export GOOGLE_API_KEY="your-api-key"
     ```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your Excel file using the sidebar
2. Wait for the "✅ Ready to Chat!" message
3. Ask questions about your data in natural language
4. Get AI-powered responses with specific values from your data

## Example Questions

- "What are the total sales for each region?"
- "Who are the top 5 customers by revenue?"
- "What's the trend in monthly expenses?"
- "Show me the highest and lowest values in column X"
- "Calculate the average of column Y grouped by column Z"

## Requirements

- Python 3.8+
- Streamlit
- LlamaIndex
- Google Generative AI
- Pandas
- OpenPyXL

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini Pro](https://ai.google.dev/)
- Uses [LlamaIndex](https://www.llamaindex.ai/) for document processing
