# Generative AI Application for Automated Code Generation

This repository contains the source code for a **Generative AI** application designed to generate executable code based on user prompts. The application leverages multiple AI models to solve Python programming tasks, including models from OpenAI, Google Gemini, and Hugging Face (Meta LLaMA). The project is connected to a research paper exploring the automation of software development using AI.

## Features

- **Multiple AI Models**: Supports interaction with OpenAI GPT models, Google Gemini models, and Meta LLaMA models via Hugging Face.
- **API Integration**: Easily integrates with official APIs, automatically handles requests, and logs responses.
- **Cost Tracking**: Tracks the token usage and costs associated with generating each code snippet.
- **Task Automation**: Provides a set of Python programming tasks for AI models to solve.
- **Detailed Logging**: Tracks API requests, responses, and the overall performance of each model.

## Supported Models and Endpoints

The application supports the following AI models:

### OpenAI Models
- **GPT-4 Turbo**
  - API Endpoint: `https://api.openai.com/v1/chat/completions`
- **GPT-4o Mini**
  - API Endpoint: `https://api.openai.com/v1/chat/completions`
- **GPT-3.5 Turbo**
  - API Endpoint: `https://api.openai.com/v1/chat/completions`

### Google Gemini Models
- **Gemini 1.0 Pro**
  - API Endpoint: via [Google Generative AI Python Client](https://ai.google.dev/api)
- **Gemini 1.5 Flash**
  - API Endpoint: via [Google Generative AI Python Client](https://ai.google.dev/api)
- **Gemini 1.5 Pro**
  - API Endpoint: via [Google Generative AI Python Client](https://ai.google.dev/api)

### Meta LLaMA Models via Hugging Face
- **Meta LLaMA 3.0 8B Instruct**
  - API Endpoint: `https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct`
- **Meta LLaMA 3.1 8B Instruct**
  - API Endpoint: `https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct`

## Example Tasks

The application sends the following 10 Python tasks to each AI model:

1. **Hello World Program**:
   - "Write a simple program that prints the text 'Hello, World!' in Python."
2. **Sum of Numbers**:
   - "Write a function that takes a list of numbers and returns their sum in Python."
3. **Factorial Calculation**:
   - "Write a function that calculates the factorial of a given number using recursion in Python."
4. **Palindrome Checker**:
   - "Write a function that checks if a given string is a palindrome in Python."
5. **Fibonacci Sequence**:
   - "Write a function that returns a list of the first n numbers in the Fibonacci sequence in Python."
6. **File Reading and Writing**:
   - "Write a program that reads data from a text file and writes it to a new file, numbering each line in Python."
7. **Basic Data Analysis**:
   - "Write a program that loads a dataset (CSV file) using the pandas library and performs basic statistical analysis (mean, median, standard deviation) on columns containing numerical data in Python."
8. **Web Scraping**:
   - "Write a script that downloads an HTML page from a given URL and extracts all links (tags `<a>`) using the BeautifulSoup library in Python."
9. **Data Visualization**:
   - "Write a program that loads a dataset (CSV file) using pandas and creates graphs (e.g., bar chart, histogram) using the Matplotlib or Seaborn library in Python."
10. **Basic Web Application**:
   - "Write a basic web application using the Flask framework that has a simple form for entering text and displays the entered text back on the page upon form submission in Python."

## Results

The generated code, response time, token usage, and cost for each task and model will be saved to `api_test_results.csv`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or further information, please contact:

- **Author**: Ing. Dominik palla
- **Email**: dominik.palla@uhk.cz