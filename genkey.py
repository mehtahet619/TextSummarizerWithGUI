import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import pipeline
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress specific warning related to tokenization spaces
warnings.filterwarnings("ignore", category=FutureWarning, message="`clean_up_tokenization_spaces` was not set")

# Initialize the summarizer
summarizer = pipeline('summarization', model="facebook/bart-large-cnn", framework="pt")

class SummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer with Accuracy")

        # Label for input text
        self.label = tk.Label(root, text="Enter text to summarize:")
        self.label.pack(pady=10)

        # Text area for user input
        self.text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=60)
        self.text_input.pack(pady=5)

        # Button to generate summary
        self.summarize_button = tk.Button(root, text="Summarize", command=self.summarize_text)
        self.summarize_button.pack(pady=5)

        # Text area for displaying summary
        self.text_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, width=60, state=tk.DISABLED)
        self.text_output.pack(pady=10)

        # Label for accuracy
        self.accuracy_label = tk.Label(root, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=5)

    def summarize_text(self):
        # Get the text from the input area
        input_text = self.text_input.get("1.0", tk.END).strip()

        if not input_text:
            messagebox.showwarning("Input Error", "Please enter some text to summarize.")
            return

        # Dynamically set max_length based on input length
        input_length = len(input_text)
        max_length = max(15, input_length // 2)  # Choose a max_length that makes sense

        try:
            # Generate summary
            summary = summarizer(input_text, max_length=max_length, min_length=5, do_sample=False)
            summary_text = summary[0]['summary_text']

            # Display the summary
            self.text_output.config(state=tk.NORMAL)
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, summary_text)
            self.text_output.config(state=tk.DISABLED)

            # Calculate accuracy (similarity score as a proxy)
            accuracy = self.calculate_accuracy(input_text, summary_text)
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def calculate_accuracy(self, original_text, summary_text):
        # Vectorize the original text and summary using TF-IDF
        vectorizer = TfidfVectorizer().fit_transform([original_text, summary_text])
        vectors = vectorizer.toarray()

        # Compute cosine similarity
        cosine_sim = cosine_similarity(vectors)
        similarity_score = cosine_sim[0][1]  # Similarity between the original text and summary

        # Convert similarity score to percentage for readability
        return similarity_score * 100

if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizerApp(root)
    root.mainloop()
