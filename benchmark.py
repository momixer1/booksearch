from engine import get_results
import numpy as np

def mrr(results:list, target: int):
    for book in results:
        # print(f"Id: {book['Id']}; Rank: {book['rank']}")
        if book["Id"] == target:
            return 1/book["rank"]
    return 0

mrr_score = []
queries = [["Harry Potter's first year at Hogwarts", 77523], 
            ["heartbreaking romance", 24850],
            ["Ein verstörter Jugendlicher wandert durch New York und sucht nach Echtheit", 14641],
            ["Tiere rebellieren gegen Menschen, doch die Tyrannei kehrt in neuer Form zurück", 7613],
            ["Zwei verfeindete Familien führen ihre jugendlichen Liebenden in den gemeinsamen Tod", 6823],
            ["A future fireman burns illegal books until he questions society", 4384],
            ["A brave girl journeys to the Arctic to save kidnapped children", 18115],
            ["A young man stays youthful while his hidden portrait ages and decays", 9121],
            ["An aging Cuban fisherman struggles relentlessly with a giant marlin", 2172],
            ["小男孩Max大鬧一場後，航向怪獸出沒的島嶼並當上國王", 19546]]


for i in range(len(queries)):
    mrr_score.append(mrr(get_results(queries[i][0], 10), queries[i][1]))
print(mrr_score)
print(np.mean(mrr_score))