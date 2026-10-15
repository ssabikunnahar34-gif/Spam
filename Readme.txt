
In _new, by oversampling spam in the training set to a 50/50 ratio, 
we forced the SVM model to give equal importance to spam patterns.

Pros: The model became more sensitive to detecting spam, 
catching 2 extra spam messages that _old missed 
(Recall increased to 90.84%).

Cons: Because it became more aggressive at catching spam, 
it flagged  additional borderline ham messages as spam 

Since accuracy only decreased from 98% to 97%, catching more spams
seems like a good improvement
