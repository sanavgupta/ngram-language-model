import pandas as pd
import numpy as np
import re
import requests
import time

# --- 1. Data Processing ---

def get_book(url):
    """Fetches book text from Project Gutenberg."""
    try:
        time.sleep(0.5)
    except:
        pass
    r = requests.get(url)
    text = r.text.replace('\r\n', '\n')
    start = re.search(r"\*\*\* START OF", text)
    end = re.search(r"\*\*\* END OF", text)
    if start and end:
        return text[start.end():end.start()]
    return text

def tokenize(text):
    """Splits text into tokens with Start/Stop markers."""
    text = text.strip()
    if not text: return ['\x02', '\x03']
    tokens = []
    for para in re.split(r'\n{2,}', text):
        if not para.strip(): continue
        tokens.append('\x02')
        tokens.extend(re.findall(r'\w+|[^\w\s]', para))
        tokens.append('\x03')
    return tokens

# --- 2. Language Models ---

class UnigramLM:
    def __init__(self, tokens):
        counts = pd.Series(tokens).value_counts()
        self.probs = counts / counts.sum()
    
    def sample(self, M):
        tokens = np.random.choice(self.probs.index, size=M, p=self.probs.values)
        return ' '.join(tokens)

class NGramLM:
    def __init__(self, N, tokens):
        self.N = N
        self.ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens)-N+1)]
        self.mdl = self._train(self.ngrams)
        if N > 2: self.prev = NGramLM(N-1, tokens)
        else: self.prev = UnigramLM(tokens)

    def _train(self, ngrams):
        counts = {}
        history_counts = {}
        for ng in ngrams:
            counts[ng] = counts.get(ng, 0) + 1
            hist = ng[:-1]
            history_counts[hist] = history_counts.get(hist, 0) + 1
        
        rows = []
        for ng, count in counts.items():
            prob = count / history_counts[ng[:-1]]
            rows.append({'ngram': ng, 'history': ng[:-1], 'prob': prob})
        return pd.DataFrame(rows)

    def sample(self, M):
        output = ['\x02']
        for _ in range(M):
            if len(output) >= self.N: 
                context = tuple(output[-(self.N-1):])
            else:
                context = tuple(output)
            
            candidates = self.mdl[self.mdl['history'] == context]

            if candidates.empty: 
                output.append('\x03')
                break
                
            next_tok = np.random.choice(
                candidates['ngram'].apply(lambda x: x[-1]), 
                p=candidates['prob']
            )
            output.append(next_tok)
        return ' '.join(output[1:])

# --- 3. Main Execution ---

if __name__ == "__main__":
    print("Fetching Shakespeare...")
    text = get_book('https://www.gutenberg.org/files/1524/1524-0.txt') 
    tokens = tokenize(text)
    
    print(f"Training on {len(tokens)} tokens...")
    model = NGramLM(3, tokens)
    
    print("\nGenerated Shakespeare (3-gram):")
    print(model.sample(50))
