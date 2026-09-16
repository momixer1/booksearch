from engine import get_results

while True:
    print('-' * 50)
    results = get_results(input("What type of book are you looking for? "), 10)
    print('-' * 50)
    for book in results:
        print(f"{book['rank']}: {book['title']} | rating: {(book['rating'])} | score: {book['score']}%")