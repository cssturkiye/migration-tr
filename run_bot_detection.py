#!/usr/bin/env python
"""
Migration-TR Bot Detection Inference Script
-------------------------------------------

This script provides bot detection capabilities for Twitter/X users based on
profile characteristics and behavioral patterns.

Example Usage
-------------
$ python run_bot_detection.py --features user_data.json
$ python run_bot_detection.py --single-user "{'usr': 'example_user', ...}"

Features Required:
- account_age, statuses_count, followers_count, friends_count, listed_count
- verified, followers_friends_ratio, statuses_followers_ratio
- screen_name_length, num_digits_in_screen_name, name_length, num_digits_in_name
- description_length, has_desc_url, location, hour_created, network
"""

import argparse
import json
import sys
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union
import onnxruntime as rt

class FeatureExtractor:
    """Extract bot detection features from user data"""
    
    def __init__(self, user_data: Dict):
        """
        Initialize feature extractor with user data.
        
        Args:
            user_data: Dictionary containing user profile information
        """
        self.user_data = user_data
        
    def extract_features(self) -> np.ndarray:
        """Extract all features required for bot detection"""
        
        # Calculate account age
        account_age = self.calc_user_age(
            self.user_data.get("usrLastTweetDate", datetime.now()),
            self.user_data.get("usrCreated", datetime.now())
        )
        
        # Basic counts
        statuses_count = self.user_data.get("usrStatusesCount", 0)
        followers_count = self.user_data.get("usrFollowersCount", 0)
        friends_count = self.user_data.get("usrFriendsCount", 0)
        listed_count = self.user_data.get("usrListedCount", 0)
        
        # Verification status
        verified = 1 if self.user_data.get("usrVerified", False) else 0
        
        # Ratios
        followers_friends_ratio = followers_count / (friends_count + 1)
        statuses_followers_ratio = statuses_count / (followers_count + 1)
        
        # Screen name features
        screen_name = self.user_data.get("usr", "")
        screen_name_length = len(screen_name)
        num_digits_in_screen_name = self.count_numerical_chars(screen_name)
        
        # Display name features
        name = self.user_data.get("usrDn", "")
        name_length = len(name)
        num_digits_in_name = self.count_numerical_chars(name)
        
        # Description features
        description = self.user_data.get("usrDes", "")
        description_length = len(description)
        
        # URL in description
        has_desc_url = 1 if self.user_data.get("usrDesLinks") else 0
        
        # Location
        location = 1 if self.user_data.get("usrLocation") else 0
        
        # Account creation hour
        created_at = self.user_data.get("usrCreated", datetime.now())
        hour_created = created_at.hour if hasattr(created_at, 'hour') else 0
        
        # Network measure
        network = np.log(1 + statuses_count) * np.log(1 + followers_count)
        
        # Construct feature vector
        features = np.array([
            account_age,
            statuses_count,
            followers_count,
            friends_count,
            listed_count,
            verified,
            followers_friends_ratio,
            statuses_followers_ratio,
            screen_name_length,
            num_digits_in_screen_name,
            name_length,
            num_digits_in_name,
            description_length,
            has_desc_url,
            location,
            hour_created,
            network
        ], dtype=np.float32)
        
        return features
    
    def count_numerical_chars(self, string: str) -> int:
        """Count numerical characters in a string"""
        return sum(1 for char in string if char.isnumeric())
    
    def calc_user_age(self, last_tweet_time, user_creation_time) -> float:
        """Calculate user account age in days"""
        if isinstance(last_tweet_time, str):
            last_tweet_time = datetime.fromisoformat(last_tweet_time.replace('Z', '+00:00'))
        if isinstance(user_creation_time, str):
            user_creation_time = datetime.fromisoformat(user_creation_time.replace('Z', '+00:00'))
            
        return (last_tweet_time - user_creation_time).total_seconds() / 86400

class BotDetector:
    """Bot detection using trained XGBoost model"""
    
    def __init__(self, model_path: str = "trained_models/bot_clf/pipeline_xgboost_wo_rates.onnx"):
        """
        Initialize bot detector with ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = Path(model_path)
        self.session = None
        self.load_model()
        
    def load_model(self):
        """Load the ONNX model for inference"""
        try:
            self.session = rt.InferenceSession(str(self.model_path))
            print(f"✅ Bot detection model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Predict if user is a bot based on features.
        
        Args:
            features: Feature vector for the user
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Reshape features for single prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Run inference
            pred_onx = self.session.run(None, {"input": features})
            
            # Extract results
            prediction = pred_onx[0][0]  # Class prediction (0 or 1)
            probabilities = pred_onx[1][0]  # Probability dict
            
            # Get bot probability
            bot_probability = probabilities.get(1, 0.0)
            
            return {
                "is_bot": bool(prediction),
                "bot_probability": float(bot_probability),
                "human_probability": float(probabilities.get(0, 0.0)),
                "prediction_class": int(prediction)
            }
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

def load_user_data(file_path: str) -> Union[Dict, List[Dict]]:
    """Load user data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Failed to load user data: {e}")
        raise

def process_single_user(user_data: Dict, detector: BotDetector) -> Dict:
    """Process a single user for bot detection"""
    try:
        # Extract features
        extractor = FeatureExtractor(user_data)
        features = extractor.extract_features()
        
        # Predict
        result = detector.predict(features)
        
        # Add user info
        result["user_id"] = user_data.get("usrID", "unknown")
        result["username"] = user_data.get("usr", "unknown")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing user: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def process_batch_users(users_data: List[Dict], detector: BotDetector) -> List[Dict]:
    """Process multiple users for bot detection"""
    results = []
    
    for i, user_data in enumerate(users_data):
        print(f"Processing user {i+1}/{len(users_data)}...")
        result = process_single_user(user_data, detector)
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Migration-TR Bot Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bot_detection.py --features users.json
  python run_bot_detection.py --single-user '{"usr": "example", "usrCreated": "2020-01-01", ...}'
  python run_bot_detection.py --features users.json --output results.json
        """
    )
    
    parser.add_argument(
        "--features", "-f",
        type=str,
        help="Path to JSON file containing user data"
    )
    
    parser.add_argument(
        "--single-user", "-u",
        type=str,
        help="JSON string of single user data"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default="trained_models/bot_clf/pipeline_xgboost_wo_rates.onnx",
        help="Path to ONNX model file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.features and not args.single_user:
        parser.error("Either --features or --single-user must be provided")
    
    try:
        # Initialize detector
        detector = BotDetector(args.model_path)
        
        # Process input
        if args.single_user:
            # Single user from command line
            user_data = json.loads(args.single_user)
            results = process_single_user(user_data, detector)
            
        else:
            # Multiple users from file
            users_data = load_user_data(args.features)
            
            if isinstance(users_data, dict):
                # Single user in file
                results = process_single_user(users_data, detector)
            else:
                # Multiple users in file
                results = process_batch_users(users_data, detector)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


"""
# Example Usage
# Single user from command line
python run_bot_detection.py --single-user '{"usr": "example", "usrCreated": "2020-01-01", ...}'

# Multiple users from file
python run_bot_detection.py --features users.json --output results.json
"""
if __name__ == "__main__":
    main()