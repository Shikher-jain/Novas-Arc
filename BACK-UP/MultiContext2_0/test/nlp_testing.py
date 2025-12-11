from shared_config import analyze_query

def analyse_user_query(user_query: str):
    context = analyze_query(user_query)
    return context

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    result = analyse_user_query(user_query)
    # print(result)
    print(vars(result))  # Print all fields as a dictionary
