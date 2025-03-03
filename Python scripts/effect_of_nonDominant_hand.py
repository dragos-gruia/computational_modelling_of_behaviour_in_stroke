

def plot_group_handImpairment_effect(path_files, output_path, dv, demographic_file, patient_group=None,  labels_dv = None, column_mapping=None, random_effects=['ID','timepoint'], independent_vars = ['age','gender','english_secondLanguage','education_Alevels','education_bachelors','education_postBachelors', 'NIHSS at admission or 2 hours after thrombectomy/thrombolysis', 'timepoint', 'impaired_hand'], var_of_interest = 'impaired_hand'):
    
    warnings.simplefilter("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)
    
    fig, ax = plt.subplots(int(len(path_files)/3), 3, figsize=(25, 8*len(path_files)/3))  # Adjust the figure size if necessary
    fig.subplots_adjust(hspace=0.2)
    
    # Define the ranges for the two values
    first_range = range(6)  # 0 to 5
    second_range = range(3) # 0 to 2
    # Generate all possible combinations (Cartesian product)
    subplot_coordinates = list(itertools.product(first_range, second_range))

    i=0
    
    for task_path in path_files:
        
        effSize = []
        placeholder = []
        pvalues = []
        panel_number= subplot_coordinates[i]
        
        # Load task data
        df = pd.read_csv(task_path)
        df = df.reset_index(drop=True)
        
        task_name = task_path.split('/')[-1].split('_outcomes')[0]
        
        # Merge demographic characteristics with the AS and DT metrics
        os.chdir('/'.join(task_path.split('/')[0:-1]))
        
        if demographic_file.split('.')[-1] == 'csv':
            df_dem = pd.read_csv(f'../trial_data/{demographic_file}')
        elif demographic_file.split('.')[-1] == 'xlsx':
            df_dem = pd.read_excel(f'../trial_data/{demographic_file}')
        else:
            print('Incorrect demgographic file names')
            return 0
        
        df_motor = get_motor_information('../trial_data/')
        df = df.merge(df_dem, how='left', on='user_id').merge(df_motor, how='left', on='user_id')
        
        # Assign raw primary outcome to 'Accuracy' column
        df.loc[:,'NIHSS at admission or 2 hours after thrombectomy/thrombolysis'] = np.log(df['NIHSS at admission or 2 hours after thrombectomy/thrombolysis']+1)
        df['age'] = (df['age'] - df['age'].mean())/df['age'].std()

        if (task_name != 'IC3_rs_SRT') & (task_name !='IC3_NVtrailMaking'):
            df['Accuracy'] = df[task_name]  

        df = df.dropna(subset=dv).reset_index(drop=True)
        df = df.dropna(subset=independent_vars).reset_index(drop=True)
        
        if patient_group == 'acute':
            df = df[df.timepoint == 1].reset_index(drop=True)
        elif patient_group == 'chronic':
            df = df[df.timepoint != 1].reset_index(drop=True)
            
        X = df.loc[:,independent_vars]
        
        df["Intercept"] = 1
        
        for dv_measure in dv:
            
            Y = df[dv_measure]
            Y = (Y - Y.mean())/Y.std()

            # Fit the regression model
            if random_effects != None:
                if len(random_effects) == 1:
                    model = sm.MixedLM(Y,X, groups = df[random_effects[0]])
                elif len(random_effects) == 2:
                    model = sm.MixedLM(endog=Y,exog=X, groups = df[random_effects[0]], exog_re=df[['Intercept',random_effects[1]]].copy())
            else:
                model = sm.OLS(Y,X)
                
            results = model.fit()
            
            # Get the predictor coefficient
            beta = results.params[var_of_interest]
            pvalue = results.pvalues[var_of_interest]
            # Calculate standardized beta
            predictor_std = X[var_of_interest].std()
            residual_std = np.sqrt(results.scale)
            standardized_beta = beta * (predictor_std / residual_std)
            effSize.append(np.abs(standardized_beta))
            placeholder.append('')
            pvalues.append(pvalue)

        _,pvalues = fdrcorrection(pvalues,alpha=0.05, method='indep')
        pvalues = np.round(pvalues,2)

        # Plot the effect of hand impairement

        custom_colors = ['#ffd11a', '#62428a', '#ffd11a', '#62428a']

        sn.barplot(y=effSize, x=placeholder, palette=custom_colors, hue=dv, ax=ax[panel_number])
        
        c=0
        for bar in ax[panel_number].patches:
            if c>=len(custom_colors)/2:
                bar.set_hatch('//')  # Possible hatches: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
            c+=1
            
        ax[panel_number].axhline(y=0, linestyle='-', color='black', linewidth=1)
        ax[panel_number].set_ylim(-0.1,0.8)
        
        if labels_dv != None:
            handles, labels =  ax[panel_number].get_legend_handles_labels()
            new_handles = handles[::1]  # Get only the handles for left and right
            ax[panel_number].legend(new_handles, labels_dv)
            ax[panel_number].legend(new_handles, labels_dv, fontsize=12)
        
        if column_mapping != None:
            ax[panel_number].text(0.5, 1.1, column_mapping[task_name], transform=ax[panel_number].transAxes, fontsize=20, va='top', ha='center', bbox = dict(boxstyle='round', facecolor = 'white'))
        else:
            ax[panel_number].text(0.5, 1.15, task_name, transform=ax[panel_number].transAxes, fontsize=20, va='top', ha='center')
           
        if panel_number[1] == 0:
            ax[panel_number].set_ylabel('Standardised beta coefficient', fontsize=16)
        else:
            ax[panel_number].set_ylabel('')

        fig.legend().remove()
        counter = 0
        for p in ax[panel_number].patches:
            
            # get the height of each bar
            height = 0 if p.get_height() != p.get_height() else p.get_height()
            height = np.max(height)
            
            # adding text to each bar
            ax[panel_number].text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar      
            y = height+0.02 if height>0 else height-0.01, # y-coordinate position of data label, padded 100 above bar
            s = "{:.2f}".format(height) if pvalues[counter]>=0.05 else "{:.2f}*".format(height), # data label, formatted to ignore decimals
            ha = "center",
            fontweight= 'regular' if pvalues[counter]>=0.05 else 'semibold',
            fontsize= 16) # sets horizontal alignment (ha) to center
            counter = counter + 1
        
        i = i+1
    
    fig.savefig(f'{output_path}/handImpairment_effects.png', format='png', transparent=False)
    