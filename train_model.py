"""
SpamShield - Model Training Script
====================================
This script trains and evaluates the spam detection ML model.
Run this script ONCE to train and save the model.

Usage:
    python train_model.py
"""

import os
import sys
import pickle

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from model import SpamDetectionModel
from sklearn.model_selection import train_test_split


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("SPAMSHIELD - EMAIL SPAM DETECTION MODEL TRAINING")
    print("=" * 70 + "\n")
    
    try:
        # Configuration
        data_path = "data/spam.csv"
        model_path = "trained_model.pkl"
        vectorizer_path = "vectorizer.pkl"
        selector_path = "feature_selector.pkl"
        
        # Check if model already exists
        if all(os.path.exists(p) for p in [model_path, vectorizer_path, selector_path]):
            response = input("Model already exists. Do you want to retrain? (y/n): ").strip().lower()
            if response != 'y':
                print("Using existing model. Run 'python main.py' to start the application.")
                return
        
        # Initialize model
        print("📊 Initializing ML Model Pipeline...")
        model = SpamDetectionModel(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            selector_path=selector_path,
        )
        
        # 1. Load data
        print("\n1️⃣ LOADING DATA")
        print("-" * 70)
        X, y = model.load_data(data_path)
        
        # 2. Split data
        print("\n2️⃣ TRAIN-TEST SPLIT")
        print("-" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"✅ Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # 3. Preprocess data
        print("\n3️⃣ DATA PREPROCESSING")
        print("-" * 70)
        X_train_tfidf = model.preprocess_data(X_train)
        X_test_tfidf = model.vectorizer.transform(X_test)
        print("✅ Preprocessing complete")
        
        # 4. Feature selection
        print("\n4️⃣ FEATURE SELECTION")
        print("-" * 70)
        X_train_selected = model.select_features(X_train_tfidf, y_train, n_features=2000)
        X_test_selected = model.feature_selector.transform(X_test_tfidf)
        print("✅ Feature selection complete")
        
        # 5. Train model
        print("\n5️⃣ MODEL TRAINING")
        print("-" * 70)
        model.train_model(X_train_selected, y_train)
        print("✅ Model training complete")
        
        # 6. Evaluate model
        print("\n6️⃣ MODEL EVALUATION")
        print("-" * 70)
        metrics = model.evaluate_model(X_test_selected, y_test)
        print("✅ Model evaluation complete")
        
        # 7. Save model
        print("\n7️⃣ SAVING MODEL")
        print("-" * 70)
        model.save_model()
        
        # Save feature selector
        os.makedirs(os.path.dirname(selector_path) or '.', exist_ok=True)
        with open(selector_path, 'wb') as f:
            pickle.dump(model.feature_selector, f)
        print(f"✅ Feature selector saved to {selector_path}")
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ MODEL TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Performance Summary:")
        print(f"   • Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   • Precision: {metrics['precision']:.2%}")
        print(f"   • Recall:    {metrics['recall']:.2%}")
        print(f"   • F1-Score:  {metrics['f1_score']:.2%}")
        
        print(f"\n📁 Model Files Saved:")
        print(f"   • {model_path}")
        print(f"   • {vectorizer_path}")
        print(f"   • {selector_path}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run: python main.py")
        print(f"   2. Open: http://127.0.0.1:5000")
        print(f"   3. Paste message text to check for spam\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"Make sure the data file exists at: {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
