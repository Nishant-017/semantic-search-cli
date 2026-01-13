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

#x . y = 3*1 + 2*0 + 0*0 + 5*0 = 3
#||x|| = √ (3)^2 + (2)^2 + (0)^2 + (5)^2 = 6.16
#||y|| = √ (1)^2 + (0)^2 + (0)^2 + (0)^2 = 1
#(x, y) = 3 / (6.16 * 1) = 0.49
 
def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    if a.shape!=b.shape:
        raise ValueError("Vectors must be of the same dimensions.")
    
    dot_prod = np.dot(a,b)
    mag_a= np.sqrt(np.sum(a**2))
    mag_b= np.sqrt(np.sum(b**2))

    if mag_a==0 or mag_b==0:
        raise ValueError("Cannot compute cosine similarity for zero magnitude vector.") 

    return float (dot_prod/(mag_a*mag_b))

#print(cosine_similarity([3,2,0,5],[0,0,0,0]))  # test



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
        return "Not similar"    

print(interpret_score(0.72)) 