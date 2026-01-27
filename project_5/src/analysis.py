import scipy.stats as st

def hypothesis_test(df_relevant):
    # alpha = 0.05
    #—Average user ratings of the Xbox One and PC platforms are the same.
    xbox_one_query ="platform == 'XOne'"
    pc_query =  "platform == 'PC'"
    xbox_one_scores = df_relevant.query(xbox_one_query)['user_score']
    pc_scores = df_relevant.query(pc_query)['user_score']

    xbox_one_vs_pc_result = st.ttest_ind(xbox_one_scores, pc_scores, nan_policy='omit', equal_var=False)
    print(f'xbox_one_vs_pc_result p-value: {xbox_one_vs_pc_result.pvalue}\n the mean scores are not the same - reject H0')


    #Average user ratings for the Action and Sports genres are the same.
    action_query = 'genre == "Action"'
    sports_query = 'genre == "Sports"'
    action_scores = df_relevant.query(action_query)['user_score']
    sports_scores = df_relevant.query(sports_query)['user_score']

    action_vs_sports_results = st.ttest_ind(action_scores, sports_scores, nan_policy='omit', equal_var=False)
    print(f'action_vs_sports_results p-value: {action_vs_sports_results.pvalue}\n the mean scores are the same - accept H0')