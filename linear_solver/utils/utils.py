import numpy as np
from scipy.sparse import csr_matrix

def criterioDiArresto(r, b, tol, it, maxIter):
    if np.linalg.norm(r)/ b < tol:
        return False
    if it >= maxIter:
        
        return False
    return True


def has_zero_in_diagonal(U: np.ndarray):
    len = U.shape[0]
    for i in range(len):
        if abs(U[i][i]) < 1e-10:
            print("Determinante é zero")
            return True
    return False



def info_matrice(A):
    # Assicurati che A sia CSR o CSC per usare la sparsità efficacemente
    if not isinstance(A, csr_matrix):
        A = A.tocsr()  # Converti in CSR se non lo è già

    dominanza = True
    for i in range(A.shape[0]):
        # Ottieni il valore diagonale
        diag = abs(A[i, i])
        
        # Somma degli elementi fuori diagonale per la riga i
        row = A.getrow(i).toarray().flatten()  # Convertiamo la riga in array denso
        off_diag_sum = np.sum(np.abs(row)) - diag  # Somma degli altri elementi

        # Se la diagonale non è maggiore della somma degli altri, non c'è dominanza diagonale
        if diag < off_diag_sum:
            dominanza = False
    A = A.toarray()
    # Restituisce i dati della matrice
    return {
        "condizionamento": np.round(np.linalg.cond(A)),  # Convertiamo in array denso per il calcolo
        "simmetria": np.allclose(A, A.T),  # Controllo simmetria per matrici sparse
        "positività": np.all(np.linalg.eigvals(A) > 0),  # Positività (su matrice densa)
        "dominanza": dominanza,
    }

def chiedi_esame_proprieta():
    while True:
        risposta = input("Vuoi esaminare le proprietà delle matrici (operazione costosa)? (S/N): ").strip().lower()  # .lower() per permettere 's' o 'S'
        if risposta == 's':
            return True
        elif risposta == 'n':
            return False
        else:
            print("Risposta non valida. Per favore, rispondi con 'S' o 'N'.")

def num_iterazioni():
    while True:
        iterazioni = input("Selezionare un numero di iterazioni massime: ")
        try:
            iterazioni = int(iterazioni)  # Prova a convertire in intero
            return iterazioni  # Se va a buon fine, ritorna l'intero
        except ValueError:
            print("Scegliere un numero intero valido.")  # Se c'è un errore, chiede di riprovare




    
def calculate_error(U, x, b):
    return np.linalg.norm(b - np.dot(U, x)) / np.linalg.norm(b)
