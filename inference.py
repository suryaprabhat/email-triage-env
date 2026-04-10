import os
import json
import argparse
from typing import Dict, Any
from email_triage_env import EmailTriageEnv

class EmailTriageBaseline:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.use_mock = False
        
        # Override with environment variables injected by OpenEnv LiteLLM proxy
        actual_api_key = os.environ.get("API_KEY") or api_key
        base_url = os.environ.get("API_BASE_URL")
        self.model = os.environ.get("MODEL_NAME") or model
        
        if not actual_api_key or actual_api_key.startswith("sk-your-") or actual_api_key == "sk-...":
            print("Warning: No valid API key provided. Using mock baseline.", flush=True)
            self.use_mock = True
            return

        try:
            from openai import OpenAI
            client_args = {"api_key": actual_api_key}
            if base_url:
                client_args["base_url"] = base_url
            self.client = OpenAI(**client_args)
        except ImportError:
            print("Warning: openai not installed. Using mock.", flush=True)
            self.use_mock = True
            
    def classify_email(self, sender: str, subject: str, body: str) -> Dict[str, str]:
        if self.use_mock:
            # A simple deterministic mock based on keyword matching
            text = f"{subject} {body}".lower()
            category = "other"
            severity = "low"
            
            if "invoice" in text or "payment" in text:
                category = "billing"
                severity = "medium"
            elif "password" in text or "login attempt" in text:
                category = "security"
                severity = "urgent"
            elif "log in" in text or "bug" in text:
                category = "support"
                severity = "high"
            elif "million" in text or "prize" in text or "scam" in sender.lower():
                category = "spam"
                severity = "low"
            elif "feature" in text or "api" in text:
                category = "product"
                severity = "low"
                
            return {"category": category, "severity": severity}

        prompt = f"""
        Classify the following email into one of these categories: [spam, support, billing, product, security, other]
        And assign a severity: [urgent, high, medium, low]
        
        Sender: {sender}
        Subject: {subject}
        Body: {body}
        
        Return ONLY valid JSON in this format:
        {{"category": "category", "severity": "severity"}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate email categorization assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "category": result.get("category", "other").lower(),
                "severity": result.get("severity", "low").lower()
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {"category": "other", "severity": "low"}

    def run_evaluation(self, num_emails: int, seed: int, output_file: str = "results.json"):
        env = EmailTriageEnv(num_emails=num_emails, seed=seed)
        obs, info = env.reset()
        
        results = {
            "num_emails": num_emails,
            "classifications": [],
            "metrics": {},
            "api_calls": 0
        }
        
        task_name = "email-triage"
        print(f"[START] task={task_name}", flush=True)
        print(f"Starting evaluation of {num_emails} emails...")
        
        done = False
        step_count = 0
        while not done:
            step_count += 1
            print(f"Processing email {obs.email_id}...")
            
            classification = self.classify_email(obs.sender, obs.subject, obs.body)
            if not self.use_mock:
                results["api_calls"] += 1
                
            action = {
                "email_id": obs.email_id,
                "category": classification["category"],
                "severity": classification["severity"]
            }
            
            results["classifications"].append({
                "email_id": obs.email_id,
                "predicted_category": action["category"],
                "predicted_severity": action["severity"]
            })
            
            obs, reward, done, _, info = env.step(action)
            print(f"[STEP] step={step_count} reward={float(reward)}", flush=True)
            
        stats = info.get("stats", {})
        total = env.num_emails
        
        cat_acc = (stats.get("correct_categories", 0) / total) * 100 if total > 0 else 0
        sev_acc = (stats.get("correct_severities", 0) / total) * 100 if total > 0 else 0
        
        detected_spam = stats.get("total_spam", 0) - stats.get("spam_missed", 0)
        spam_detect_rate = (detected_spam / stats.get("total_spam", 1)) * 100 if stats.get("total_spam", 0) > 0 else 100.0
        
        final_score = info.get("current_score", 0)
        results["metrics"] = {
            "accuracy_category": cat_acc,
            "accuracy_severity": sev_acc,
            "accuracy_both": (cat_acc + sev_acc) / 2,
            "spam_detection_rate": spam_detect_rate,
            "final_score": final_score,
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Evaluation complete. Results saved to {output_file}")
        print(f"Final Score: {final_score}")
        print(f"[END] task={task_name} score={final_score} steps={step_count}", flush=True)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline inference for Email Triage")
    parser.add_argument("--num-emails", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4-turbo-preview")
    parser.add_argument("--output", type=str, default="results.json")
    
    args = parser.parse_args()
    api_key = os.environ.get("HF_TOKEN", "")
    
    baseline = EmailTriageBaseline(api_key=api_key, model=args.model)
    baseline.run_evaluation(args.num_emails, args.seed, args.output)
