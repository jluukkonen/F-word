import requests
import time
import re
import pandas as pd
from datetime import datetime
import os

# Configuration
SEARCH_REGEX = r"\bfuck[s(ing)(ed)]?\b"
SUBREDDITS = {
    "weak-tie": ["AskReddit", "funny", "pics", "worldnews", "todayilearned"],
    "strong-tie": ["programming", "MechanicalKeyboards", "linux", "guitars", "Parenting"]
}
BASE_URL = "https://api.pullpush.io/reddit/search/comment/"
PARENT_URL = "https://api.pullpush.io/reddit/search/comment/?ids="
OUTPUT_FILE = "data/raw/reddit_fuck.tsv"
FETCH_PER_SUBREDDIT = 50  # Total desired per subreddit for the pilot
DELAY = 3.0

def fetch_comments(subreddit, network_tie, limit=50):
    comments_found = []
    before = int(time.time())
    
    print(f"Collecting from r/{subreddit} ({network_tie})...")
    
    while len(comments_found) < limit:
        params = {
            "subreddit": subreddit,
            "q": "fuck",
            "size": 100,
            "before": before,
            "sort": "desc"
        }
        
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            
            if not data:
                print(f"  No more data found for r/{subreddit}.")
                break
                
            for comment in data:
                text = comment.get("body", "")
                if re.search(SEARCH_REGEX, text, re.IGNORECASE):
                    comment_id = comment.get("id")
                    parent_id = comment.get("parent_id")
                    
                    # Store basic info, we'll fetch parents in a batch or one by one
                    comments_found.append({
                        "id": comment_id,
                        "text": text,
                        "parent_id": parent_id,
                        "subreddit": subreddit,
                        "network_tie_strength": network_tie
                    })
                    
                    if len(comments_found) >= limit:
                        break
            
            # Update 'before' for pagination
            before = data[-1].get("created_utc")
            print(f"  Found {len(comments_found)} comments so far...")
            
            time.sleep(DELAY)
            
        except Exception as e:
            print(f"  Error fetching from r/{subreddit}: {e}")
            break
            
    return comments_found

def fetch_parent_texts(comments):
    print("\nFetching parent comment texts...")
    # To be efficient, we'd batch this, but for a showcase, we can do it simply.
    # PullPush ids parameter can take a comma-separated list.
    
    ids_to_fetch = [c["parent_id"].split("_")[-1] for c in comments if c["parent_id"] and "_" in c["parent_id"]]
    parent_map = {}
    
    # Process in chunks of 50
    for i in range(0, len(ids_to_fetch), 50):
        chunk = ids_to_fetch[i:i+50]
        url = PARENT_URL + ",".join(chunk)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json().get("data", [])
            for p in data:
                parent_map[p["id"]] = p.get("body", "[No text or deleted]")
            
            time.sleep(DELAY)
        except Exception as e:
            print(f"  Error fetching parents: {e}")
            
    # Update comments with parent text
    for c in comments:
        pid = c["parent_id"].split("_")[-1] if "_" in c["parent_id"] else c["parent_id"]
        c["parent_text"] = parent_map.get(pid, "[Context Not Found]")
        # Remove parent_id from final output
        del c["parent_id"]
        
    return comments

def main():
    all_data = []
    
    for tie_type, subs in SUBREDDITS.items():
        for sub in subs:
            data = fetch_comments(sub, tie_type, limit=FETCH_PER_SUBREDDIT)
            all_data.extend(data)
            
    # Fetch context
    full_data = fetch_parent_texts(all_data)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(full_data)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved {len(df)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
