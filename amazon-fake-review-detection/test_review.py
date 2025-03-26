import argparse
from predict import predict_review

def main():
    parser = argparse.ArgumentParser(description='Test a review for authenticity')
    parser.add_argument('--review', type=str, required=True, help='The review text to analyze')
    parser.add_argument('--title', type=str, default='', help='Review title (optional)')
    parser.add_argument('--rating', type=float, default=None, help='Rating (optional)')
    parser.add_argument('--verified', type=str, default=None, choices=['Y', 'N'], 
                        help='Verified purchase (Y/N, optional)')
    
    args = parser.parse_args()
    
    print("\n===== REVIEW AUTHENTICITY ANALYSIS =====")
    print(f"Review: {args.review}")
    if args.title:
        print(f"Title: {args.title}")
    if args.rating:
        print(f"Rating: {args.rating}")
    if args.verified:
        print(f"Verified Purchase: {args.verified}")
    print("="*40)
    
    results = predict_review(
        review_text=args.review,
        review_title=args.title,
        rating=args.rating,
        verified_purchase=args.verified
    )
    
    print("\nPrediction Results:")
    print("-"*20)
    for model_name, result in results.items():
        prediction = result['prediction']
        confidence = result['confidence']
        
        confidence_str = f" (Confidence: {confidence:.2%})" if confidence is not None else ""
        print(f"{model_name}: {prediction}{confidence_str}")
    
    # Calculate ensemble prediction (majority vote)
    predictions = [result['prediction'] for result in results.values()]
    fake_count = predictions.count('FAKE')
    genuine_count = predictions.count('GENUINE')
    
    print("-"*20)
    if fake_count > genuine_count:
        print(f"OVERALL: LIKELY FAKE ({fake_count}/{len(predictions)} models)")
    elif genuine_count > fake_count:
        print(f"OVERALL: LIKELY GENUINE ({genuine_count}/{len(predictions)} models)")
    else:
        print("OVERALL: UNCERTAIN (models disagree)")
    print("="*40)

if __name__ == "__main__":
    main()
