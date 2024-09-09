import time
import pandas as pd
import logging
import google.generativeai as genai
import requests
import tiktoken
from transformers import AutoTokenizer

# Configure logging to display more information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API keys (replace these with your actual API keys)
OPENAI_API_KEY = 'API_KEY'
GOOGLE_GEMINI_API_KEY = 'API_KEY'
HUGGINGFACE_API_KEY = 'API_KEY'

# Universal instruction to append to all prompts
UNIVERSAL_INSTRUCTION = " Give me only the executable code, no comments, no explanations, no other information, just the code."

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Function to calculate token count using TikToken (OpenAI tokenizer)
def calculate_tokens(text, model="gpt-4-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_tokens_transformers(text, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.encode(text))
    except Exception as e:
        logging.error(f"Error with tokenizing using Hugging Face AutoTokenizer: {e}")
        return 0


# OpenAI API function with input/output cost calculation (returns only total cost)
def get_openai_code(prompt, model="gpt-4-turbo", input_cost_per_token=0.01, output_cost_per_token=0.03):
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 150,
        'temperature': 0.7
    }

    logging.info(f"Sending request to OpenAI chat model {model}...")

    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=data)
    end_time = time.time()

    logging.info(f"Full API response from OpenAI: {response.text}")
    response_data = response.json()

    if 'choices' in response_data:
        generated_code = response_data['choices'][0]['message']['content']
    else:
        generated_code = None

    # Calculate input and output tokens
    input_tokens = calculate_tokens(prompt, model)
    output_tokens = calculate_tokens(generated_code, model)

    # Calculate the total cost based on input and output tokens
    total_cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

    logging.info(f"Response from OpenAI (model: {model}) received in {end_time - start_time} seconds.")
    logging.info(f"Total cost: ${total_cost}")
    return generated_code, end_time - start_time, total_cost


# Google Gemini API function with cost calculation (returns only total cost)
def get_gemini_code(prompt, model="gemini-1.5-flash", input_cost_per_token=0.0015, output_cost_per_token=0.0025):
    logging.info(f"Sending request to Google Gemini model {model}...")

    start_time = time.time()

    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        generated_code = response.text

        # Calculate input and output tokens
        input_tokens = calculate_tokens(prompt)
        output_tokens = calculate_tokens(generated_code)

        # Calculate the total cost based on input and output tokens
        total_cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

        logging.info(f"Total cost: ${total_cost}")
    except Exception as e:
        logging.error(f"Error while using Google Gemini API: {str(e)}")
        return None, None, None

    end_time = time.time()

    logging.info(f"Response from Google Gemini (model: {model}) received in {end_time - start_time} seconds.")
    return generated_code, end_time - start_time, total_cost

# Meta LLaMA API function (Hugging Face) with total cost calculation based on input cost
def get_llama_code(prompt, model="meta-llama/Meta-Llama-3-8B-Instruct", input_cost_per_token=0.002, output_cost_per_token=0.002):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'inputs': prompt,
        'parameters': {
            'temperature': 0.7,
            'max_new_tokens': 150
        }
    }

    logging.info(f"Sending request to Hugging Face model {model}...")

    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=data)
    end_time = time.time()

    logging.info(f"Full API response from Hugging Face: {response.text}")

    try:
        response_data = response.json()

        if 'error' in response_data:
            logging.error(f"Error from Hugging Face: {response_data['error']}")
            return None, None, None

        generated_text = response_data[0].get('generated_text', "")

        # Check if code block markers exist
        if "```" in generated_text:
            generated_code = generated_text.split("```")[1].strip()
        else:
            generated_code = generated_text.strip()

        # Calculate the total tokens used (prompt + generated code)
        total_tokens = calculate_tokens_transformers(prompt + generated_code, model)

        # Use only input cost as total cost (ignore output cost)
        total_cost = total_tokens * input_cost_per_token

        logging.info(f"Generated code: {generated_code}")
        logging.info(f"Total tokens used: {total_tokens}, Total cost: ${total_cost}")
        return generated_code, end_time - start_time, total_cost
    except Exception as e:
        logging.error(f"Error parsing JSON response from Hugging Face: {e}")
        return None, None, None


# Define tasks with prompts and universal instruction
tasks = [
    "Write a simple program that prints the text 'Hello, World!' in Python." + UNIVERSAL_INSTRUCTION,
    "Write a function that takes a list of numbers and returns their sum in Python." + UNIVERSAL_INSTRUCTION,
    "Write a function that calculates the factorial of a given number using recursion in Python." + UNIVERSAL_INSTRUCTION,
    "Write a function that checks if a given string is a palindrome in Python." + UNIVERSAL_INSTRUCTION,
    "Write a function that returns a list of the first n numbers in the Fibonacci sequence in Python." + UNIVERSAL_INSTRUCTION,
    "Write a program that reads data from a text file and writes it to a new file, numbering each line in Python." + UNIVERSAL_INSTRUCTION,
    "Write a program that loads a dataset (CSV file) using the pandas library and performs basic statistical analysis (mean, median, standard deviation) on columns containing numerical data in Python." + UNIVERSAL_INSTRUCTION,
    "Write a script that downloads an HTML page from a given URL and extracts all links (tags <a>) using the BeautifulSoup library in Python." + UNIVERSAL_INSTRUCTION,
    "Write a program that loads a dataset (CSV file) using pandas and creates graphs (e.g., bar chart, histogram) using the Matplotlib or Seaborn library in Python." + UNIVERSAL_INSTRUCTION,
    "Write a basic web application using the Flask framework that has a simple form for entering text and displays the entered text back on the page upon form submission in Python." + UNIVERSAL_INSTRUCTION
]

# List of models to test with cost per token
models = [
    {"name": "OpenAI GPT-4 Turbo", "function": get_openai_code, "model_param": "gpt-4-turbo",
    "input_cost_per_token": 0.01/1000, "output_cost_per_token": 0.03/1000},
    {"name": "OpenAI GPT-4o", "function": get_openai_code, "model_param": "gpt-4o",
    "input_cost_per_token": 0.005/1000, "output_cost_per_token": 0.015/1000},
    {"name": "OpenAI GPT-4o Mini", "function": get_openai_code, "model_param": "gpt-4o-mini",
    "input_cost_per_token": 0.00015/1000, "output_cost_per_token": 0.0006/1000},
    {"name": "OpenAI GPT-3.5 Turbo", "function": get_openai_code, "model_param": "gpt-3.5-turbo",
    "input_cost_per_token": 0.003/1000, "output_cost_per_token": 0.006/1000},
    {"name": "Google Gemini 1.0 Pro", "function": get_gemini_code, "model_param": "gemini-1.0-pro",
     "input_cost_per_token": 0.0005/1000, "output_cost_per_token": 0.0015/1000},
    {"name": "Google Gemini 1.5 Flash", "function": get_gemini_code, "model_param": "gemini-1.5-flash",
     "input_cost_per_token": 0.000075/1000, "output_cost_per_token": 0.0003/1000},
    {"name": "Google Gemini 1.5 Pro", "function": get_gemini_code, "model_param": "gemini-1.5-pro",
     "input_cost_per_token": 0.0035/1000, "output_cost_per_token": 0.0105/1000},
    {"name": "Meta LLaMA 3.0 8B", "function": get_llama_code, "model_param": "meta-llama/Meta-Llama-3-8B-Instruct",
    "input_cost_per_token": 0.0028/1000, "output_cost_per_token": 0.0028/1000},
    {"name": "Meta LLaMA 3.1 8B", "function": get_llama_code, "model_param": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "input_cost_per_token": 0.0028/1000, "output_cost_per_token": 0.0028/1000},
]

# Initialize list to store results
results = []

# Testing loop
for task in tasks:
    logging.info(f"Testing task: {task}")
    for model in models:
        for iteration in range(3):  # Run 3 iterations
            logging.info(f"Testing model: {model['name']} (Iteration {iteration + 1})")
            try:
                # Call the function for all models with input/output costs
                result_text, response_time, total_cost = model['function'](
                    task,
                    model['model_param'],
                    model['input_cost_per_token'],
                    model['output_cost_per_token']
                )

                logging.info(f"Result for {model['name']} (Iteration {iteration + 1}):")
                logging.info(f"Response Time: {response_time} seconds, Total Cost: ${total_cost}")
                logging.info(f"Generated Code: {result_text}")

                # Append results to list
                results.append({
                    "Model": model['name'],
                    "Task": task,
                    "Iteration": iteration + 1,
                    "Response Time (s)": response_time,
                    "Generated Code": result_text,
                    "API Cost (USD)": total_cost
                })
            except Exception as e:
                logging.error(f"Error with model {model['name']} on task: {task} - {str(e)}")
                results.append({
                    "Model": model['name'],
                    "Task": task,
                    "Iteration": iteration + 1,
                    "Response Time (s)": None,
                    "Generated Code": None,
                    "API Cost (USD)": None,
                    "Error": str(e)
                })
# Convert results to DataFrame and save as CSV
df = pd.DataFrame(results)
df.to_csv("api_test_results.csv", index=False)

logging.info("Testing complete. Results saved to api_test_results.csv")