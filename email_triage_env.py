import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum
import random
import time

class EmailCategory(str, Enum):
    spam = "spam"
    support = "support"
    billing = "billing"
    product = "product"
    security = "security"
    other = "other"

class EmailSeverity(str, Enum):
    urgent = "urgent"
    high = "high"
    medium = "medium"
    low = "low"

class Observation(BaseModel):
    email_id: int
    sender: str
    subject: str
    body: str
    timestamp: str
    processed: bool
    current_category: Optional[str] = None
    current_severity: Optional[str] = None
    emails_processed: int
    total_emails: int
    score: float

class Action(BaseModel):
    email_id: int
    category: str
    severity: str
    notes: Optional[str] = None

class EmailGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.templates = [
            {"subject": "Your invoice #12345 is ready", "body": "Please find attached your invoice for this month's services.", "sender": "billing@company.com", "category": "billing", "severity": "medium"},
            {"subject": "URGENT: Password Reset Requested", "body": "A password reset was requested for your account. If this wasn't you, please secure your account immediately.", "sender": "security@company.com", "category": "security", "severity": "urgent"},
            {"subject": "Help with my account", "body": "I can't seem to log in since yesterday. Please help.", "sender": "user@gmail.com", "category": "support", "severity": "high"},
            {"subject": "You won a million dollars!!", "body": "Click here to claim your prize now and become rich!!!", "sender": "scam@scammer.com", "category": "spam", "severity": "low"},
            {"subject": "Feedback on your new feature", "body": "I really like the new dashboard, but it would be great if I could export to CSV.", "sender": "client@enterprise.com", "category": "product", "severity": "low"},
            {"subject": "Lunch today?", "body": "Hey team, are we still doing lunch at 12?", "sender": "bob@company.com", "category": "other", "severity": "low"},
            {"subject": "Action Required: API Deprecation", "body": "We are deprecating the v1 API next month. Please migrate to v2.", "sender": "api@provider.com", "category": "product", "severity": "high"},
            {"subject": "Suspicious login attempt", "body": "We noticed a login from an unrecognized device in Antarctica.", "sender": "alerts@security.com", "category": "security", "severity": "urgent"},
            {"subject": "Payment failed", "body": "Your recent payment failed. Please update your card details.", "sender": "billing@company.com", "category": "billing", "severity": "high"},
            {"subject": "Bug report: UI glitch", "body": "The sidebar disappears on mobile devices.", "sender": "user@gmail.com", "category": "support", "severity": "medium"}
        ]
        
    def generate(self, num_emails: int) -> List[Dict]:
        emails = []
        for i in range(num_emails):
            template = self.rng.choice(self.templates)
            email = {
                "email_id": i + 1,
                "sender": template["sender"],
                "subject": template["subject"],
                "body": template["body"],
                "timestamp": f"2024-04-08T10:{self.rng.randint(10, 59):02d}:00Z",
                "ground_truth_category": template["category"],
                "ground_truth_severity": template["severity"],
            }
            emails.append(email)
        return emails

class EmailTriageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_emails: int = 10, seed: Optional[int] = None):
        super().__init__()
        self.num_emails = num_emails
        self._seed = seed
        self.generator = EmailGenerator(seed=self._seed)
        
        self.action_space = spaces.Dict({
            "email_id": spaces.Discrete(100000),
            "category": spaces.Text(max_length=20),
            "severity": spaces.Text(max_length=20),
            "notes": spaces.Text(max_length=1000)
        })
        
        self.observation_space = spaces.Dict({
            "email_id": spaces.Discrete(100000),
            "sender": spaces.Text(max_length=100),
            "subject": spaces.Text(max_length=200),
            "body": spaces.Text(max_length=2000),
            "timestamp": spaces.Text(max_length=50)
        })
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Observation, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.generator = EmailGenerator(seed=seed)
            self._seed = seed
            
        self.emails = self.generator.generate(self.num_emails)
        self.current_idx = 0
        self.score = 0.0
        self.stats = {
            "correct_categories": 0,
            "correct_severities": 0,
            "spam_missed": 0,
            "total_spam": sum(1 for e in self.emails if e["ground_truth_category"] == "spam")
        }
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> Observation:
        if self.current_idx >= len(self.emails):
            # Return dummy if done
            return Observation(
                email_id=-1, sender="", subject="", body="", timestamp="",
                processed=True, emails_processed=self.num_emails, total_emails=self.num_emails,
                score=self.score
            )
            
        email = self.emails[self.current_idx]
        return Observation(
            email_id=email["email_id"],
            sender=email["sender"],
            subject=email["subject"],
            body=email["body"],
            timestamp=email["timestamp"],
            processed=False,
            emails_processed=self.current_idx,
            total_emails=self.num_emails,
            score=self.score
        )

    def _get_info(self) -> dict:
        accuracy = 0
        if self.current_idx > 0:
            accuracy = ((self.stats["correct_categories"] + self.stats["correct_severities"]) / 
                       (2 * self.current_idx)) * 100
                       
        return {
            "current_score": self.score,
            "accuracy": accuracy,
            "emails_remaining": self.num_emails - self.current_idx,
            "stats": self.stats
        }

    def step(self, action: Union[Dict, Action]) -> Tuple[Observation, float, bool, bool, dict]:
        if self.current_idx >= len(self.emails):
            return self._get_obs(), 0.0, True, False, self._get_info()
            
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except ValidationError as e:
                # Invalid action penalty
                self.score -= 0.5
                return self._get_obs(), -0.5, False, False, {"error": str(e)}

        current_email = self.emails[self.current_idx]
        
        # Verify action is for current email
        if action.email_id != current_email["email_id"]:
            return self._get_obs(), -0.1, False, False, {"error": "Email ID mismatch"}

        # Calculate reward
        reward = 0.0
        cat_correct = (action.category == current_email["ground_truth_category"])
        sev_correct = (action.severity == current_email["ground_truth_severity"])
        
        if cat_correct:
            reward += 0.5
            self.stats["correct_categories"] += 1
        if sev_correct:
            reward += 0.5
            self.stats["correct_severities"] += 1
            
        # Efficiency simple simulation
        reward += 0.1
        
        # Safety penalty for spam
        if current_email["ground_truth_category"] == "spam" and not cat_correct:
            reward -= 0.3
            self.stats["spam_missed"] += 1

        self.score += reward
        self.current_idx += 1
        
        # Adjust score precision to avoid float issues
        self.score = round(self.score, 2)
        reward = round(reward, 2)
        
        done = self.current_idx >= len(self.emails)
        
        return self._get_obs(), float(reward), done, False, self._get_info()

    def render(self):
        obs = self._get_obs()
        print(f"--- Email {obs.email_id}/{self.num_emails} ---")
        print(f"From: {obs.sender}")
        print(f"Subject: {obs.subject}")
        print(f"Preview: {obs.body[:50]}...")
        print(f"Current Score: {obs.score:.2f}")
        print("-------------------")

    def get_stats(self) -> Dict[str, Any]:
        return self.stats
