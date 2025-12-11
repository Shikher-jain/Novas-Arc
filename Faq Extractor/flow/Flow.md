This script is a web crawler designed to extract **FAQ-style content** from a given website, process it, and save it in a structured format. Let's break down the flow of the script step by step.

---

### 1. **`main()` Function**

The **entry point** of the program that drives the entire crawling process.

* **Inputs:**

  * URL of the website to crawl.
  * Depth of crawl (how many levels of links to follow from the starting URL).
  * Max workers (number of threads for parallel processing).
  * Min answer length for filtering FAQs.
* **Outputs:**

  * Saves two JSONL files containing FAQ data: one for generic Q&A (simple question-answer pairs), and another optimized for fine-tuning AI models in OpenAI chat format.

#### Flow:

* **Prompts the user** for the starting URL, crawl depth (default 3), max worker threads (default 12).
* **Generates site name** from the URL's domain for file naming.
* **Creates folders** (`QnA`, `FineTuning`) if they don't exist.
* **Checks for existing files**: If both QnA and FineTuning JSONL files already exist for the site, skips extraction and informs the user.
* If files don't exist, **starts the crawling process** using `crawl_site()`.
* After crawling, **prompts for min answer length** (default 20) and filters FAQs accordingly.
* **Saves filtered FAQs** to both formats: simple Q&A in `QnA/{site}.jsonl` and chat-format in `FineTuning/{site}.jsonl`.
* Prints crawl time, pages crawled, FAQs extracted, and file paths.

---

### 2. **Crawl Site (`crawl_site()`)**

The function that performs the actual crawling of the website to collect all relevant URLs and FAQ data.

#### Flow:

* Initializes **seen URLs set** with thread-safe lock to avoid revisiting pages.
* Starts with the **root URL** in the frontier and crawls up to the specified **depth**.
* For each depth level, processes the current frontier in parallel using ThreadPoolExecutor.
* Uses **breadth-first search** (BFS) approach: processes all pages at current depth before moving to next.

#### Key tasks:

1. **Deduplicates frontier** to avoid processing already-seen URLs.
2. **Submits each URL** in the batch to `process_page()` for parallel processing.
3. **Collects results**: URLs processed, FAQs extracted, and new links discovered.
4. **Builds next frontier** from discovered links, filtering for same domain and unseen URLs.
5. **Deduplicates final FAQs** by question (case-insensitive) and ensures minimum length.
6. Returns list of crawled URLs and deduplicated FAQs.

---

### 3. **Process Page (`process_page()`)**

This function processes each individual page to extract FAQ data and discover relevant internal links.

#### Flow:

* **Fetches HTML** content using `fetch_url()`; returns empty if failed.
* **Extracts FAQ data** from HTML using `extract_faqs_from_html()`.
* **Extracts relevant links** using `extract_links()`, filtering for same domain and FAQ keywords.
* **Follows placeholder links** in FAQ answers (e.g., "click here" links) to ensure comprehensive coverage.
* **Deduplicates links** and returns the list of links and extracted FAQs.

---

### 4. **Extract FAQ Data (`extract_faqs_from_html()`)**

This function performs comprehensive extraction of FAQ questions and answers from HTML content, handling various common formats.

#### Flow:

* **Removes noise elements** (scripts, styles, nav, footer, etc.) from the soup.
* Extracts FAQs using multiple strategies:

  * **JSON-LD structured data** (SEO-optimized FAQ pages).
  * **Accordion items** (combined: accordion-item, accordion, AWS expandable sections).
  * **Details/Summary** accordion elements.
  * **Definition lists** (dl/dt/dd).
  * **Headings** (h2/h3/h4) that look like questions.
  * **Tables** (Q in first column, A in second).
  * **Q:/A:** patterns in text blocks.
  * **Section-based FAQs** (h3 questions with following content).
  * **Flexible accordion handling** (additional patterns).
  * **Topic grids** (specific to some sites).
* **Cleans extracted text** using `clean_text()` for questions and `clean_answer()` for answers.
* **Deduplicates FAQs** within the page by question and minimum length.

---

### 5. **Extract Links (`extract_links()`)**

This function finds and filters relevant links in the page that are likely to lead to additional FAQ content.

#### Flow:

* Parses all anchor tags from the HTML.
* **Normalizes URLs** to absolute form.
* **Filters for same domain** only.
* **Excludes static assets** (.pdf, .jpg, .css, .js, etc.) and tracking URLs.
* **Applies keyword filter** using FAQ_KEYWORDS_IN_URL (faq, help, support, etc.).
* **Deduplicates** the final link list.

---

### 6. **Clean FAQ Data (`clean_text()`, `clean_answer()`, `remove_abb()`)`

These helper functions clean and preprocess FAQ data to ensure quality and consistency.

* **`clean_text()`**: Removes unwanted characters, leading numbers/Q prefixes, duplicate sentences, expands abbreviations.
* **`clean_answer()`**: Trims answers that contain new questions or known question texts.
* **`remove_abb()`**: Expands contractions (don't → do not) using the contractions library.

---

### 7. **Save the Data (`save_faqs_jsonl()`)`

Saves FAQ data in JSONL format optimized for fine-tuning AI models.

#### Flow:

* For each FAQ, creates a **chat-format record** with:
  * System message: "You are a helpful assistant."
  * User message: The question.
  * Assistant message: The answer.
* Writes each record as a JSON object on a new line.

---

### 8. **Main Loop in `crawl_site()`**

The crawling uses a level-by-level BFS approach:

* **Initial frontier**: Contains only the root URL.
* **Depth iteration**: For each depth level (0 to max_depth):
  * Process current frontier URLs in parallel.
  * Collect new links for next frontier.
  * Deduplicate and filter next frontier.
* **Final deduplication**: FAQs deduplicated across all pages.

---

### 9. **Fine-Tuning Preparation**

* **Length filtering**: After crawling, FAQs are filtered by minimum answer length.
* **Format conversion**: 
  * **QnA folder**: Simple JSONL with {"question": "...", "answer": "..."} objects.
  * **FineTuning folder**: Chat-format JSONL using `save_faqs_jsonl()` for OpenAI fine-tuning.

---

### 10. **Final Output**

After completion:

* Prints **crawl duration**, **pages crawled**, **FAQs extracted**.
* Confirms **file save locations** for both QnA and FineTuning files.
* Optional: Launches Streamlit app for Q&A interaction (via main.py).

---

### **Summary**

* **Input**: A URL of a website, crawl depth (default 3), max workers (default 12), min answer length (default 20).
* **Process**:

  * Check if FAQ data already exists; skip if files present.
  * Crawl through the website using BFS up to the specified depth, processing pages in parallel.
  * Extract FAQ data from various HTML formats and follow relevant internal links.
  * Clean, deduplicate, and filter FAQs by answer length.
  * Save results as two JSONL files: simple Q&A pairs and OpenAI chat-format for fine-tuning.
* **Output**: Detailed logs on crawl time, pages crawled, FAQs extracted, and file locations. Optional Streamlit Q&A interface.

---

### Example Use Case:

* To extract FAQ data from `example.com`, enter the URL, set depth/workers if needed, and specify min answer length. The script crawls the site, extracts Q&A from FAQ sections in multiple formats, cleans the data, and saves it for AI fine-tuning or direct Q&A use. Optionally, launch a Streamlit app to query the extracted FAQs.
