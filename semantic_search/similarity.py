import numpy as np

def dot_product(a,b):

    a=np.array(a)
    b=np.array(b)

    if a.shape != b.shape:
       raise ValueError("Vectors must be of the same dimensions for dot product.")
    
    return float (np.dot(a,b))

#print(dot_product([1,2,3],[4,5,6]))  # Example


#Euclidean: sqrt(sum((A-B)^2))

def euclidean_distance(a,b):
    a=np.array(a)
    b=np.array(b)

    if a.shape != b.shape:
       raise ValueError("Vectors must be of the same dimensions for Euclidean distance.")
    
    return float (np.sqrt(np.sum((a-b)**2)))

#print(euclidean_distance([8,3],[4,6]))  # test


#Cosine: dot(A,B) / (norm(A) * norm(B))
 
def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    if a.shape!=b.shape:
        raise ValueError("Vectors must be of the same dimensions.")
    
    dot_prod = np.dot(a,b)
    mag_a= np.sqrt(np.sum(a**2))
    mag_b= np.sqrt(np.sum(b**2))

    if mag_a==0 or mag_b==0:
        return 0.0

    return float (dot_prod/(mag_a*mag_b))

#print(cosine_similarity([3,2,0,5],[0,0,0,0]))  # test

# function to intepret the similarity score

def interpret_score(score):
   
    if score >= 0.9:
        return "Extremely similar"
    elif score >= 0.7:
        return "Very similar"
    elif score >= 0.5:
        return "Similar"
    elif score >= 0.3:
        return "Somewhat similar"
    elif score >= 0.1:
        return "Weakly similar"
    else:
        return "Not similar at all"    
    
#print (cosine_similarity([8,3],[4,6]))



#find top-k similar embeddings function

def find_top_k(query_emb, corpus_embs, k=5):
        
    scores = []

    # 1) compare query with every corpus embedding
    for i, doc_emb in enumerate(corpus_embs):
        score = cosine_similarity(query_emb, doc_emb)
        scores.append((i, score))  # store (index, score)

    # 2) sort by score 

    def get_score(item):
        return item[1]  # item = (index, score)
    
    scores = sorted(scores, key=get_score, reverse=True)


    # 3) return only top k
    return scores[:k]
