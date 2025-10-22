import numpy as np #per vettori e matrici 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler #per standardizzare i dati
from sklearn.datasets import load_svmlight_file
from time import process_time_ns #per misurare il tempo di esecuzione
from sklearn.svm import SVC #per SVM di sklearn


class FrankWolfeSVM:
    """
    SVM lineare allenata con Frank–Wolfe sul problema duale.

    Parametri
    ---------
    C : float
        Vincolo a scatola sul duale (0 <= alpha_i <= C).
    max_iter : int
        Numero massimo di iterazioni FW.
    tol : float
        Soglia di arresto sul *dual gap*.
    """

    def __init__(self, C, max_iter=1000, tol=1e-6): # inizializza i parametri
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        # Attributi settati in fit
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.w = None
        self.history = None

    # ------------------------------
    # Calcolo Qv
    # ------------------------------
    def _Q_dot_linear(self, v): # v = alpha o direzione d
        """Calcola Qv in modo efficiente, senza calcolare Q
        Qui Q = (yy^T) ⊙ (XX^T). Calcolo Qv, ovvero Q * alpha oppure Q * d.:
        z = y ⊙ v,
        u = X^T z,
        t = X u,
        Qv = y ⊙ t.
        """
        z = self.y_train * v
        u = self.X_train.T @ z
        t = self.X_train @ u
        return self.y_train * t

    def _update_w_from_alpha(self):
        """Aggiorna w = X^T (alpha ⊙ y) """
        self.w = self.X_train.T @ (self.alpha * self.y_train)

    def _f_no_b(self):
        """Restituisce i punteggi f(x) senza bias sui dati di train."""
        return self.X_train @ self.w

    # ------------------------------
    # Calcolo obbiettivo e gradiente
    # ------------------------------
    def compute_dual_objective(self, alpha):
        """Valore dell'obiettivo duale: 1^T alpha - 0.5 * ||w||^2 con w = X^T (alpha ⊙ y)"""
        w = self.X_train.T @ (alpha * self.y_train)
        return np.sum(alpha) - 0.5 * np.dot(w, w)

    def compute_gradient(self, alpha):
        """Gradiente di (0.5 α^T Q α - 1^T α) = Qα - 1 """
        return self._Q_dot_linear(alpha) - 1.0

    def linear_minimization_oracle(self, grad, y):
        """LMO esatto sul politopo {s: 0<=s<=C, y^T s = 0}.
        Ordina gradiente per classi e accoppia i migliori
        (uno +1 e uno -1) impostandoli a C se riduce il valore lineare.
        Risolve il sottoproblema min_s s^T grad s.t. 0<=s<=C, y^T s = 0.
        Returns
        -------
        s : np.ndarray
            Soluzione ottima del LMO.
        """
        s = np.zeros_like(grad)
        # separazioni indici pos e neg
        pos = np.where(y == 1)[0]
        neg = np.where(y == -1)[0]
        if pos.size == 0 or neg.size == 0:
            return s
        # ordina indici per valore crescente di gradiente
        pos_sorted = pos[np.argsort(grad[pos])]
        neg_sorted = neg[np.argsort(grad[neg])]
        m = min(pos_sorted.size, neg_sorted.size)
        for k in range(m): 
            i = pos_sorted[k]
            j = neg_sorted[k]
            if grad[i] + grad[j] < 0.0: # se grad[i] + grad[j] < 0 allora assegna s[i]=C e s[j]=C
                s[i] = self.C
                s[j] = self.C
            else:
                break
        return s

    def line_search(self, alpha, d):
        """
        Calcola lo step gamma FW ottimo su [0,1] per phi(α) = 0.5 α^T Q α - 1^T α
        Minimizza lungo la direzione d = s - α
        """
        Qa = self._Q_dot_linear(alpha)
        Qd = self._Q_dot_linear(d) # direzione d = s - alpha
        num = - (d @ Qa - np.sum(d))    
        denom = d @ Qd
        if denom <= 0:
            return 1.0
        return np.clip(num / denom, 0.0, 1.0)

    def _compute_b_from_f(self, alpha, y, f_no_b):
        """Stima del bias b.
        - Se esistono SV liberi (0<α_i<C): media di y_i - f_no_b,i 
            poichè y_i (f_no_b,i + b) = 1 --> b = y_i - f_no_b,i
        - Altrimenti usa i limiti provenienti da SV a C di classi opposte.
        """
        free = (alpha > 1e-8) & (alpha < self.C - 1e-8)
        if np.any(free):
            return np.mean(y[free] - f_no_b[free])
        pos_atC = (y == 1) & (alpha > self.C - 1e-8)
        neg_atC = (y == -1) & (alpha > self.C - 1e-8)
        b_up = np.min(1 - f_no_b[pos_atC]) if np.any(pos_atC) else np.inf
        b_low = np.max(-1 - f_no_b[neg_atC]) if np.any(neg_atC) else -np.inf
        if np.isfinite(b_up) and np.isfinite(b_low):
            return 0.5 * (b_up + b_low)
        if np.isfinite(b_up):
            return b_up
        if np.isfinite(b_low):
            return b_low
        return np.mean(y - f_no_b)

    # ------------------------------
    # Training & Inferenza
    # ------------------------------
    def fit(self, X_train, y_train, verbose=True):
        """Esegue Frank–Wolfe fino a gap < tol o raggiunto max_iter."""
        self.X_train = X_train
        self.y_train = y_train.astype(float)
        n = X_train.shape[0]
        self.alpha = np.zeros(n)
        self._update_w_from_alpha()  # w = 0 all'inizio
        self.history = {"dual_obj": [], "gap": [], "train_acc": []}

        for k in range(self.max_iter):
            grad = self.compute_gradient(self.alpha)
            s = self.linear_minimization_oracle(grad, self.y_train)
            d = s - self.alpha
            gap = grad @ (self.alpha - s) # Dual Gap

            gamma = self.line_search(self.alpha, d)
            d_alpha = gamma * d # Step size ottimo

            # Aggiorna α e w (in modo incrementale)
            self.alpha += d_alpha
            self.w += self.X_train.T @ (d_alpha * self.y_train)

            # Bookkeeping metriche
            f_tr_no_b = self._f_no_b()
            dual_obj = np.sum(self.alpha) - 0.5 * np.dot(self.w, self.w)
            b = self._compute_b_from_f(self.alpha, self.y_train, f_tr_no_b)
            train_acc = np.mean(np.sign(f_tr_no_b + b) == self.y_train)

            self.history['dual_obj'].append(dual_obj)
            self.history['gap'].append(gap)
            self.history['train_acc'].append(train_acc)

            #if verbose and (k % 1000 == 0 or k == self.max_iter - 1):
            #    print(f"[FW] iter={k:6d} gap={gap:.3e} dual={dual_obj:.6f} acc={train_acc:.4f}")

            # Early Stopping
            if gap < self.tol:
                if verbose:
                    print("Converged (gap < tol) at iteration", k)
                break

        # Stima finale del bias
        f_tr_no_b = self._f_no_b()
        self.b = self._compute_b_from_f(self.alpha, self.y_train, f_tr_no_b)
        return self

    def decision_function(self, X):
        """Restituisce f(x) = w^T x + b"""
        # check_is_fitted(self, ["alpha", "b", "X_train", "y_train", "w"])
        return X @ self.w + self.b

    def predict(self, X):
        """Etichette in {-1, +1} tramite il segno di f(x)."""
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def accuracy(self, X, y):
        """Accuratezza sulle etichette y."""
        return np.mean(self.predict(X) == y)


class SVM_SMO:
    """
    SVM lineare allenata con SMO sul duale

    Richiede in input Q = (yy^T) ⊙ (XX^T) calcolata sui dati di training.
    Parametri
    ---------
    C : float
        Vincolo a scatola sul duale (0 <= alpha_i <= C).
    tol : float
        Soglia di arresto sul gap tra errori più violanti.
    max_iter : int
        Numero massimo di iterazioni SMO.
    """

    def __init__(self, Q=None, C=1.0, tol=1e-3, max_iter=100_000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.Q = Q
        # Attributi dopo fit
        self.alpha = None
        self.b = 0.0
        self.y = None
        self.w = None  # per decision_function lineare

    def fit(self, X, y):
        """Allena SMO iterando con regola MVP e rispettando i vincoli di box e y^T α = 0."""
        self.y = y.astype(float)
        n = len(y)

        self.alpha = np.zeros(n)
        self.b = 0.0

        # g = Qα - 1, con α = 0 -> g = -1
        g = -np.ones(n)
        eps = 1e-12

        for it in range(self.max_iter):
            # Insiemi attivi (LibSVM)
            I_up = np.where(((self.alpha < self.C - eps) & (self.y == 1)) |
                            ((self.alpha > eps) & (self.y == -1)))[0]
            I_low = np.where(((self.alpha < self.C - eps) & (self.y == -1)) |
                             ((self.alpha > eps) & (self.y == 1)))[0]
            if I_up.size == 0 or I_low.size == 0:
                break

            # Errori via g (poiché g = y*f_no_b - 1)
            E = (g + 1.0) / self.y + self.b - self.y

            # Most Violating Pair (LibSVM)
            i = I_up[np.argmin(E[I_up])]     # b_up  = min Error su I_up
            j = I_low[np.argmax(E[I_low])]   # b_low = max Error su I_low
            gap = E[j] - E[i]                # = b_low - b_up

            if gap <= 2.0 * self.tol:
                print(f"Converged (gap <= 2*tol) at iter {it}.")
                break

            yi, yj = self.y[i], self.y[j]
            ai_old, aj_old = self.alpha[i], self.alpha[j]

            # Ricaviamo K da Q
            Kii = self.Q[i, i]
            Kjj = self.Q[j, j]
            Kij = yi * yj * self.Q[i, j]

            # Errori ai punti i, j (senza bias )
            f_i_no_b = (g[i] + 1.0) / yi
            f_j_no_b = (g[j] + 1.0) / yj
            Ei = (f_i_no_b + self.b) - yi
            Ej = (f_j_no_b + self.b) - yj

            # Limiti della box per l'aggiornamento 2D
            if yi != yj:
                L = max(0.0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)
            if L == H:
                continue

            # Curvatura
            eta = Kii + Kjj - 2.0 * Kij
            if eta <= 0:
                continue

            # Aggiornamento di α_j (e poi α_i per vincolo di uguaglianza)
            aj_new = aj_old + yj * (Ei - Ej) / eta
            aj_new = np.clip(aj_new, L, H)
            if abs(aj_new - aj_old) < 1e-12:
                continue
            ai_new = ai_old + yi * yj * (aj_old - aj_new)

            # Aggiornamento bias (regola a due candidati b1 o b2)
            b1 = self.b - Ei - yi * (ai_new - ai_old) * Kii - yj * (aj_new - aj_old) * Kij
            b2 = self.b - Ej - yi * (ai_new - ai_old) * Kij - yj * (aj_new - aj_old) * Kjj
            if 0 < ai_new < self.C:
                self.b = b1
            elif 0 < aj_new < self.C:
                self.b = b2
            else:
                self.b = 0.5 * (b1 + b2)

            # Aggiornamento incrementale del gradiente: g ← g + Δα_i Q[:,i] + Δα_j Q[:,j]
            d_ai = ai_new - ai_old
            d_aj = aj_new - aj_old
            if d_ai != 0.0:
                g += d_ai * self.Q[:, i]
            if d_aj != 0.0:
                g += d_aj * self.Q[:, j]

            # Commit
            self.alpha[i] = ai_new
            self.alpha[j] = aj_new

        # Rifinitura del bias via SV liberi
        f_no_b = (g + 1.0) / self.y
        free = (self.alpha > 1e-8) & (self.alpha < self.C - 1e-8)
        if np.any(free):
            self.b = np.mean(self.y[free] - f_no_b[free])
        else:
            pos = (self.y == 1) & (self.alpha > 1e-8)
            neg = (self.y == -1) & (self.alpha > 1e-8)
            b_up = np.min(1 - f_no_b[pos]) if np.any(pos) else np.inf
            b_low = np.max(-1 - f_no_b[neg]) if np.any(neg) else -np.inf
            if np.isfinite(b_up) and np.isfinite(b_low):
                self.b = 0.5 * (b_up + b_low)

        # Prepara vettore dei pesi per decision_function lineare
        self.w = X.T @ (self.alpha * self.y)
        return self

    def decision_function(self, X):
        """Distanze firmate lineari: f(x) = w^T x + b."""
        # check_is_fitted(self, ["alpha", "b", "w"])
        return X @ self.w + self.b

    def predict(self, X):
        """Etichette in {-1, +1} col segno di f(x)."""
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


# -------------------------------------------------------------
# Funzioni di utilità per il confronto tra metodi
# -------------------------------------------------------------

def compare_methods(X_train, y_train, X_test, y_test, C, Q):
    """
    Allena e valuta FW, SMO e scikit-learn SVC(lineare) sullo stesso split.

    Returns
    -------
    dict : metriche per metodo (tempo di training, acc, F1, matrici di confusione).
    """
    comparison_dict = {}
    methods = ["fw", "smo", "sklearn"]

    for method in methods:
        start_time = process_time_ns()
        print("\n" + "=" * 50)
        print(f"METHOD: {method.upper()}")
        print("=" * 50)

        if method == "fw":
            max_iter = 100_000 if C >= 100 else 10_000
            svm = FrankWolfeSVM(C=C, max_iter=max_iter, tol=1e-5)
            svm.fit(X_train, y_train, verbose=True)

        elif method == "smo":
            max_iter = 1_000_000 if C >= 100 else 100_000
            svm = SVM_SMO(Q=Q, C=C, max_iter=max_iter, tol=1e-5)
            svm.fit(X_train, y_train)

        elif method == "sklearn":
            print(f"C = {C}")
            svm = SVC(
                kernel='linear',
                C=C,
                tol=1e-5,
                shrinking=True,
                cache_size=1000,
                max_iter=-1,
            )
            svm.fit(X_train, y_train)

        # ---- Metriche uniformi ----
        finish_time = (process_time_ns() - start_time) / 1e9

        y_pred_train = svm.predict(X_train)
        y_pred_test = svm.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')

        cm_train = confusion_matrix(y_train, y_pred_train, labels=[-1, 1])
        cm_test = confusion_matrix(y_test, y_pred_test, labels=[-1, 1])

        comparison_dict[method] = {
            "training_time": finish_time,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "train_confusion_matrix": cm_train,
            "test_confusion_matrix": cm_test,
        }

        # Stampa riepilogo
        print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Training time: {finish_time:.4f} seconds")
        print(f"Training F1-score: {train_f1:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
        print("Training Confusion matrix:")
        print(cm_train)
        print("Test Confusion matrix:")
        print(cm_test)

    return comparison_dict


# -----------------------------
# MAIN: esecuzione
# -----------------------------
if __name__ == "__main__":
    datasets = ["breast_cancer_scaled", "mushroom", "australian", "diabetes", "sonar"]

    for dataset in datasets:
        X, y = load_svmlight_file(f"{dataset}.txt")
        X = X.toarray()

        # Mappa etichette su {-1, +1} dove necessario
        if dataset == "breast_cancer_scaled":
            y[y == 2] = 1
            y[y == 4] = -1
        elif dataset == "mushroom":
            y[y == 1] = 1
            y[y == 2] = -1
        y = y.astype(float)

        # Train-test split + standardizzazione (fit su train, applica a test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print("\n" + "-" * 100)
        print("DATASET:", dataset)
        print(f"Il dataset ha {X.shape[1]} feature.")

        # Precompute Q = (y y^T) ⊙ (X X^T) solo per SMO
        K_train = X_train @ X_train.T
        yyT = y_train[:, None] * y_train[None, :]
        Q = yyT * K_train

        C_values = [1e-2, 1e-1, 1, 10, 100, 1000]
        for C in C_values:
            print(f"\n------> Valore di C: {C}")
            comparison_dict = compare_methods(X_train, y_train, X_test, y_test, C, Q)
