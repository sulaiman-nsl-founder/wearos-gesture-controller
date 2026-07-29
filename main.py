import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Wear OS ML Gesture Controller Pipeline")
    parser.add_argument("action", choices=["collect", "process", "train", "run"], help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "collect":
        from src import data_collection
        data_collection.run()
    elif args.action == "process":
        from src import preprocessing
        preprocessing.run()
    elif args.action == "train":
        from src import training
        training.run()
    elif args.action == "run":
        from src import controller
        controller.run()

if __name__ == "__main__":
    main()
