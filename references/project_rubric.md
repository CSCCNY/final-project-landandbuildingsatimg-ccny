
Project Rubric and Project Grade:  
By Michael David Grossberg  
CSC I1910 DeepLearning with Tensor Flow, Spring 2021  
Project/Presentation Rubric Tips  
Read through the rubric below to know what I will look for when I grade. Note that:  

- You should be using a Data Set and apply one or more of the ML frameworks we learned in the class  
    - Multi-variable Classification
    - Multi-variable Non-linear Regression
    - Clustering
    - Dimension Reduction
    
- It is important that you attempt a real problem. Just showing relationships between variables is not enough.

- Good statistical hygiene is critical
    - Your test data is only used once for evaluation at the end never iteratively to help pick anything  
    - Hyperparamters including which model is best is done on Validation data which is separate from testing data
    - You probably should use cross-validation ONLY on the train part of a train-test so that the train part of your top-level train test split is split into ‘real’ train and validation data.
    - If you are doing regression or classification your input data must never have access to any information about the target variable. That is, once your model is trained, you should assume that you should be able to accept inputs X without Y.
    - As above, any normalization, data cleaning, dropping missing values, or outliers must assume no access to the target variable Y.
    
    - Do not spend any time explaining what you learned about machine learning in general or the class in your presentation or report. Your discussion should be at the level of explaining to a colleague who might be interested in solving this problem what useful insights about the problem and the data you can share. Negative results are fine if the experiment that found them was conducted rationally. You may also explain discoveries of missing values and garbage in the data. Things like disk failures, stories of poor time management, or misunderstandings that are unlikely to happen to others should be omitted.
    
# Baselines
You ALWAYS need to compare with simple baselines. If a coin flip, or a constant classifier, or a simple threshold on one variable beats your fancy SVM/RandomForest, the fancy stuff is worthless. Every drug the FDA approves must first beat a placebo. That is no joke. Drug companies spend hundreds of millions of dollars, possibly billions trying to beat Placebos:Placebos Are Getting More Effective. Drugmakers Are Desperate to Know Why. If your algorithm can’t beat a placebo, then you don’t have an algorithm. If you take pride in your work you won’t try and BS people. If you can’t beat a placebo you need to be straight with them.

# Balanced Classes
Both the Credit Card Fraud project and the Data Quality flag project had very unbalanced classes. When you have unbalanced classes it easily becomes, by far, the most important thing. Creutzfeldt-Jakob Disease is a rare fatal disease that effects 0.0001% of the population. Suppose you develop a test based on a machine learning classifier to detect the disease. What is the placebo test? The placebo test is the test that ALWAYS outputs “0” or negative. Such a test is “useless”. However, that test has a 99.999% accuracy. Of course, the recall of such a test is 0%. If you train an ML algorithm on unbalanced classes it will most likely learn the ALWAYS “0” useless classifier because the accuracy of this placebo is so high. Algorithms are lazy so you can’t be. The accuracy of the placebo class is always going to exactly match the prevalence of that class because they always guess 0 so the error rate if class 1 is the prevalence of class 1 (it gets all those wrong).  

Thus the recall on all the data, or the accuracy on a balanced resampled data set becomes the only meaningful measure of effectiveness. Any further analysis without taking this into account is meaningless.

# Any mistake can be “fatal”
There are some domains where you can be sloppy and you get “partial credit” for being mostly right. Often data science is not like that at all. Instead, good ideas can go very very wrong if one isn’t extremely careful. First the good idea. Berthold mentioned this idea when he pointed out that there may be something to looking at hotspots in the crime data. One has to be very careful because just because there is more crime in an area doesn’t make it a hotspot. Any place with more people is going to have more of everything, which is just like the unbalanced class we just talked about. Still, the concentration of resources where there are high amounts of crime has been an important idea many people have looked at. David Weisburd and Lawrence Sherman did some critical work which you can read about in the police experiment that changed what we know about foot patrol. There was also interesting crime algorithms developed by Hannah Fry which is discussed in Can crime be predicted by an algorithm?.

There is a big “BUT” here. When data-driven algorithms are not treated with great attention to fairness, problems like unbalanced classes, etc., instead of helping neighborhoods suffering from high crime, poverty, and racism, they can instead, make these problems much worse. A fascinating story is of the NYPD compstat system which started as a reform of a broken, corrupt system that ignored crime in poor neighborhoods letting poor residents (often people of color) fend for themselves. This first led to a better data-driven system that improved crime in many parts of New York, but then eventually became a perverse system that enforced racism and remains problematic. The fascinating saga is discussed briefly here but the amazing story is worth the listen Episode #127 The Crime Machine, Part I and here 128 The Crime Machine, Part II. Malcolm Gladwell’s book, whose starting point is the heartbreaking case of Sandra Bland, also explained how a sloppy misinterpretation of what data-driven policing was supposed to be, lead to over-aggressive over-policing. This, in turn, contributed to so many tragedies involving police and unarmed people of color talking to strangers. My point to you is that if you work on important problems with the high impact you need to get everything exactly right because any mistakes can have grave consequences.

# How I will Grade
- I will read through the commits to your group project. I will assess how much work you did relative to the rest of the class, and how much work you did with respect to your teammates on the project. I can only do this by looking at how much work I see in the notebooks (length) and the number of commits of those notebooks. I can not give you credit for thinking, reading or other activities that don’t result in something in the repo. If you must do lots of such work, I suggest you follow along by creating experiments in the notebooks or creating your own notes using markdown, inside or outside of notebooks.

- From this, I will assess a “work score”. Any code that was copied directly from the internet is not your work and does not count toward work score. In fact, anything you use externally should be properly referenced otherwise it is plagiarism.

- Your work fraction on the project = (your work score)/(sum of teams work score)

- The project grade will follow the project rubric below. Some questions below may not be relevant to some projects but all parts I-VIII are required. The total grade will be based on:

- your workscore/(mean class(work score)+2stdev(work score)) + + ((fraction of work on project) * (project grade)) + (project grade)

- The project grade will consist of the assesment of the presentation, the report, and the repository.

# Project Presentation
The project presentation will be kept to 15 minutes. I will record the presentation in class with a video recorder but I will not share that recording publicly without your permission (the recording is just for internal assessment). You should give a brief overview of the project. What the problem is. Describe the data. Describe your steps in data wrangling, data cleaning, and all the other steps in the ML pipeline. You should end with a conclusion about what someone reading your code, seeing your presentation, and reading your report should conclude from your analysis. Also, do not spend any time on what you would do if you had more time. Keep your comments restricted to what you did do.

Do not waste any time explaining what you personally learned from the process or the class during the presentation or report!

# Report
You may submit one report per group. Although the report should be done in the repo using either a jupyter notebook or markdown format. You should commit as you write so I can see who wrote what and I will take that into account when I assess the grade. The final report should not be casually written. The markdown cells should contain a coherent narrative. Every calculation should have an explanation as to why it was performed. Whenever something is calculated, there should be numbers printed out such as evaluations, or checks, or even better figures. Figures should never just be shown without explanation. The final report should stand on its own as a report which explains everything. It should not simply be a scratch book of a bunch of calculations left to an archeologist to figure out why they were done, and what one should conclude from the derived results. Sections should be labeled clearly. Comments on code should also be used to narrate what is being done and why. The final report should be a very very very cleaned up version of all the calculations done during the project. It must represent the work of the entire team, synthesized into a single report (not three reports.)

The report should roughly follow an academic paper format:

#### Title
Should express the problem in the least boring but factually accurate way. It might be interesting to ask a question.

#### Abstract
1-2 paragraphs of 200–250 words. Should concisely state the problem, why it is important, and give some indication of what you accomplished (2-3 discoveries)

#### Introduction
State your data and research question(s). Indicate why it is important. Describe your research plan so that readers can easily follow your thought process and the flow of the report. Please also include key results at the beginning so that readers know to look for. Here you can very briefly mention any important data cleaning or preparation. Do not talk about virtual results i.e. things you tried or wanted to do but didn’t do. Virtual results are worse than worthless. They highlight failure.

#### Background
Discuss other relevant work on solving this problem. Most of your references are here. Cite all sources. There is no specific formatting requirement for citations but be consistent.

#### Data
Where you go the data. Describe the variables. You can begin discussing the data wrangling, and data cleaning. Some EDA may happen here. This includes your data source (including URL if applicable), any articles behind the data source.

#### Methods
How did you take your data and set up the problem? Describe things like normalization, feature selection, the models you chose. In this section, you may have EDA and graphs showing the exploration of hyper-parameters. Note: Use graphs to illustrate interesting relationships that are important to your final analyses. DO NOT just show a bunch of graphs because you can. You should label and discuss every graph you include. There is no required number to include. The graphs should help us understand your analysis process and illuminate key features of the data.

#### Evaluation
Here you are going to show your different models’ performance. It is particularly useful to show multiple metrics and things like ROC curves (for binary classifiers). Make sure it is clearly not just what the score is but for which instances in the data one has the largest errors (in a regression), or just sample examples miss-classified. Make an attempt to interpret the parameters of the model to understand what was useful about the input data. Method comparison and sensitivity analyses are absolutely CRUCIAL to good scientific work. To that end, you MUST compare at least 2 different methods from class in answering your scientific questions. It is important to report what you tried but do so SUCCINCTLY.

#### Conclusion
How well did it work? Characterize how robust you think the results are (did you have enough data?) Try for interpretation of what the model found (what variables were useful, what was not)? Try to avoid describing what you would do if you had more time. If you have to make a statement about “future work” limit it to one short statement.

#### Attribution
Using the number and size of github commits by author (bar graph), and the git hub visualizations of when the commits occurred. Using these measures each person should self-report how many code-hours of their work are visible in the repo with 2-3 sentences listing their contribution. Do not report any code hours that cannot be traced to commits. If you spend hours on a 2-line change of code or side-reading you did, you cannot report. If you do searches or research for the project that does not result in code, you must create notes in a markdown file (eg. in the project wiki) and the notes should be commensurate with the amount of work reported. Notes cannot be simply copy-pasted from elsewhere (obviously).

#### Bibiliiography
References should appear at the end of the report/notebook. Again, no specific format is required but be consistent.

#### Appendex
If there are minor results and graphs that you think should be included, put them at the end. Do not include anything without an explanation. No random graphs just for padding!! However, let’s say you did a 50 state analysis of poverty and demographics, and your report focused on the 5 most interesting states, for completeness you could include all in an appendix. Be sure though to provide some (very short) discussion with each figure/code/result.

#### Again, this is not an opportunity to explain what you learned on the project or in the course, or express regret on mistakes


# Project Rubric
- ### I. Machine Learning Question: 20 pts
    - Is the background context for the question stated clearly (with references)?
    - Is the hypothesis/problem stated clearly ("The What")
    - Is it clear why the problems are important? Is it clear why anyone would care? ("The Why")
    - Is it clear why the data were chosen should be able to answer the question being asked?
    - How new, non-obvious, significant are your problems? Do you go beyond checking the easy and obvious?  
    
    
- ### II. Data Cleaning/Checking/Data Exploration: 20pts
    - Did you perform a thorough EDA (points below included)?
    - Did you check for outliers?
    - Did you check the units of all data points to make sure they are in the right range?
    - Did you identify the missing data code?
    - Did you reformat the data properly with each instance/observation in a row, and each variable in a column?
    - Did you keep track of all parameters and units?
    - Do you have a specific code for reformating the data that does not require information not documented (eg. magic numbers)?
    - Did you plot univariate and multivariate summaries of the data including histograms, density plots, boxplots?
    - Did you consider correlations between variables (scatterplots)?
    - Did you consider plot the data on the right scale? For example, on a log scale?
    - Did you make sure that your target variables were not contaminating your input variables?
    - If you had to make synthetic data was it a useful representation of the problem you were trying to solve?  
    
- ### III. Transformation, Feature Selection, and Modeling: 30pts
    - Did you transform, normalize, filter the data appropriately to solve your problem? Did you divide by max-min, or the sum, root-square-sum, or did you z-score the data? Did you justify what you did?
    - Did you justify normalization or lack of checking which works better as part of your hyper-parameters?
    - Did you explore univariate and multivariate feature selection? (if not why not)
    - Did you try dimension reduction and which methods did you try? (if not why not)
    - Did you include 1-2 simple models, for example with classification LDA, Logistic Regression or KNN?
    - Did you pick an appropriate set of models to solve the problem? Did you justify why these models and not others?
    - Did you try at least 4 models including one Neural Network Model using Tensor-Flow or Pytorch?
    - Did you exercise the data science models/problems we described in the lectures showing what was presented?
    - Are you using appropriate hyper-parameters? 
        - For example, if you are using a KNN regression are you investigating the choice of K and whether you use uniform or distance weighting? 
        - If you are using K-means do you explain why K? If you are using PCA do you explore how many dimensions such as by looking at the eigenvalues?  
        
- ### IV. Metrics, Validation and Evaluation 20pts
    - Are you using an appropriate choice of metrics? Are they well justified? If you are doing classification do you show a ROC curve? If you are doing regression are you justifying the metric least squares vs. mean absolute error? Do you show both?
    - Do you validate your choices of hyperparameters? For example, if you use KNN or K-means do you use cross-validation to optimize your choice of parameters?
    - Did you make sure your training and validation process never used the training data?
    - Do you estimate the uncertainty in your estimates using cross-validation?  
    - Can you say how much you are overfitting?
    
- ### V. Visualization 10pts
    - Do you provide visualization summaries for all your data and features?
    - Do you use the correct visualization type, eg. bar graphs for categorical data, scatter plots for numerical data, etc?
    - Are your axes properly labeled?
    - Do you use color properly?
    - Do you use opacity and dot size so that scatterplots with lots of data points are not just a mass of interpretable dots?
    - Do you write captions explaining what a reader should conclude from each figure (not just saying what it is but what it tells you)?
    
- ### VI. Code 20pts
    - Is the code provided can reproduce the entire work?
    - Is the data included or at least linked (externally) with instructions on how to download it?
    - Do you factor repeated operations into functions to avoid repetitively and error-prone copy-paste?
    - Do you use docstrings and numpy documentation style:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
   to make your code clear and readable?
    - Do you use markdown cells to explain every step of your code similar to Homeworks and some example notebooks?
    - Does the code demonstrate considerable work given the number of people on the project?
    
- ### VII. Presentation 30pts
    - Do you tell a coherent story with a beginning, middle, and end?
    - Do you introduce why the problem is important?
    - Do you explain in the first couple of slides what you accomplished on solving the problem?
    - Are you careful not to have slides filled with text (keep in notes)?
    - Is data and evaluations presented as clear figures (mostly)?
    - Do you make sure to say what is "interesting" or should be learned from each figure?
    - Do you stay within your time limits 15 min?
    - Do you avoid useless padding slides of no relevance?
    
- ### VIII. Report 30pts
    - The structure should follow this template above. Parts are listed in that section.




