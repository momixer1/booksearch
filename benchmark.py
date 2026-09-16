from engine import get_results
import numpy as np

def mrr(results:list, target: str):
    for book in results:
        # print(f"Id: {book['Id']}; Rank: {book['rank']}")
        if target.lower() in book["title"].lower():
            return 1/book["rank"]
    return 0

mrr_score = []
queries = [["Harry Potter's first year at Hogwarts", "Harry Potter and the Sorcerer's Stone"], 
            ["heartbreaking romance", "Scar Lover"],
            ["Ein verstörter Jugendlicher wandert durch New York und sucht nach Echtheit", "Fänger im Roggen"],
            ["Tiere rebellieren gegen Menschen, doch die Tyrannei kehrt in neuer Form zurück", "Animal Farm"],
            ["Zwei verfeindete Familien führen ihre jugendlichen Liebenden in den gemeinsamen Tod", "Romeo"],
            ["A future fireman burns illegal books until he questions society", "Fahrenheit 451"],
            ["A brave girl journeys to the Arctic to save kidnapped children", "The Golden Compass"],
            ["A young man stays youthful while his hidden portrait ages and decays", "Dorian Gray"],
            ["An aging Cuban fisherman struggles relentlessly with a giant marlin", "Old Man and the Sea"],
            ["小男孩Max大鬧一場後，航向怪獸出沒的島嶼並當上國王", "野獸國"],
            ["teenage mutants learning to control their powers at Xavier's school to become the next generation of heroes", "The New Mutants Classic"],
            ["in-depth biography of the avant-garde diarist exploring her bohemian life, relationships, and literary career", "Anaïs Nin: A Biography"],
            ["a large family with ten children moves to the country and desperately tries to convince their parents to get an animal", "Ten Kids, No Pets"],
            ["an ethical and philosophical examination of modern factory farming and the impact of our food consumption", "We Eat: Why Our Food Choices Matter"],
            ["young heroes embark on a dangerous fantasy quest across Avalon as a dark threat causes the stars to vanish", "Shadows on the Stars"],
            ["playful rhyming children's book introducing toddlers to whimsical pets and silly creatures", "Wet Pet, Dry Pet, Your Pet, My Pet"],
            ["memoir of a pioneer woman enduring hardships and the wagon journey westward during the California Gold Rush", "Frontier Lady"]]


print("evaluating...")
for i in range(len(queries)):
    mrr_score.append(mrr(get_results(queries[i][0], 10), queries[i][1]))
    # print('-' * 50)
    # results = get_results(queries[i][0], 10)
    # for book in results:
    #     print('-' * 50)
    #     print(f"{book['rank']}: {book['name']} | score: {book['score']}%")

print(mrr_score)
print(f"MRR @ 10: {np.mean(mrr_score)}")