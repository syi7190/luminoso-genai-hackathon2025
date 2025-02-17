import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import csv
import time
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=7),
    retry_error_callback=lambda _: "Error processing review. Skipping."
)
def ask_llm(prompt):
    query = PromptTemplate.from_template("""
        You are summarizing and shorten this customer review in a way where only the 
        important information is kept. Avoid using bullets, just separate things into sentences. 
        Format your answer like this:
        (Summary of customer review)
    """)
    full_prompt = f"""
        {prompt}
        Customer Review: {query}
    """
    response = model.generate_content(full_prompt)
    return response.text if response and response.text else "I apologize, I couldn't generate a complete response."

def process_reviews(input_file, output_file, batch_size=5, delay=2):
    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        print("Fieldnames:", reader.fieldnames)
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        batch = []
        processed = 0
        
        for row in reader:
            batch.append(row)
            processed += 1
            
            if len(batch) >= batch_size:
                process_batch(batch, writer)
                batch = []
                time.sleep(delay)
                print(f"Processed {processed} reviews...")
                
        if batch:
            process_batch(batch, writer)
            print(f"Processed {processed} reviews total.")

def process_batch(batch, writer):
    for row in batch:
        try:
            row['\ufeffText'] = ask_llm(row['\ufeffText'])
            writer.writerow(row)
        except Exception as e:
            print(f"Error processing review: {e}")
            writer.writerow(row)

if __name__ == "__main__":
    original_csv = "Jewelry Store Google Map Reviews.csv"
    improved_csv = "Improved Jewelry Store Google Map Reviews.csv"
    
    process_reviews(
        input_file=original_csv,
        output_file=improved_csv,
        batch_size=10,
        delay=2
    )