# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)


# Predicting Employee Attrition

This project provides a clear and practical analytics platform to help Human Resources teams understand and reduce employee attrition. Using the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-datase) from Kaggle, it uncovers the main reasons why employees leave and helps organizations take action to keep their best talent.

**Key features:**
- **Easy-to-understand Dashboards:** Six simple plots show important trends, such as how many people leave, which departments and genders are most affected, and how income, job satisfaction, and years at the company relate to attrition. These visuals are designed for everyone, whether or not you have a technical background.
- **Attrition Prediction:** The project uses machine learning to predict which employees are most likely to leave, so HR teams can act early and improve retention.

By combining clear visual analysis with predictive tools, this project turns HR data into useful insights for better workforce planning and decision-making.

## Dataset Content
* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size of 100Gb.


## Business Requirements
The main business requirements for this project are:

1. **Understand Key Drivers of Employee Attrition:** Identify the most significant factors that contribute to employees leaving the organization, such as department, gender, income, job satisfaction, and tenure.
2. **Visualize Attrition Trends:** Provide clear, actionable dashboards that allow HR and business leaders to quickly grasp attrition patterns and risk areas across the company.
3. **Predict At-Risk Employees:** Use machine learning to estimate the likelihood of individual employees leaving, enabling proactive retention strategies and targeted interventions.
4. **Support Data-Driven HR Decisions:** Deliver insights and predictions in a format that is accessible to both technical and non-technical stakeholders, supporting evidence-based workforce planning and policy development.


## Hypothesis and how to validate?

Below are five hypotheses for this project, along with the statistical test to validate each:

1. **Attrition and Department**  
	Hypothesis: The rate of employee attrition is not the same across all departments.  
	Statistical Test: Chi-square test of independence between Department and Attrition.

2. **Attrition and Gender**  
	Hypothesis: There is no significant difference in attrition rates between male and female employees.  
	Statistical Test: Chi-square test of independence between Gender and Attrition.

3. **Attrition and Monthly Income**  
	Hypothesis: Employees who leave have the same average monthly income as those who stay.  
	Statistical Test: Independent samples t-test comparing Monthly Income for Attrition = Yes vs. No.

4. **Attrition and Job Satisfaction**  
	Hypothesis: There is no difference in job satisfaction scores between employees who leave and those who stay.  
	Statistical Test: Mann-Whitney U test (if Job Satisfaction is ordinal) or t-test (if treated as continuous).

5. **Attrition and Years at Company**  
	Hypothesis: The average years at company is the same for employees who leave and those who stay.  
	Statistical Test: Independent samples t-test comparing YearsAtCompany for Attrition = Yes vs. No.

## Project Plan
* Outline the high-level steps taken for the analysis.
* How was the data managed throughout the collection, processing, analysis and interpretation steps?
* Why did you choose the research methodologies you used?

## The rationale to map the business requirements to the Data Visualisations
* List your business requirements and a rationale to map them to the Data Visualisations

## Analysis techniques used
* List the data analysis methods used and explain limitations or alternative approaches.
* How did you structure the data analysis techniques. Justify your response.
* Did the data limit you, and did you use an alternative approach to meet these challenges?
* How did you use generative AI tools to help with ideation, design thinking and code optimisation?

## Ethical considerations
* Were there any data privacy, bias or fairness issues with the data?
* How did you overcome any legal or societal issues?

## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).
* How were data insights communicated to technical and non-technical audiences?
* Explain how the dashboard was designed to communicate complex data insights to different audiences. 

## Unfixed Bugs
* Please mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation are not valid reasons to leave bugs unfixed.
* Did you recognise gaps in your knowledge, and how did you address them?
* If applicable, include evidence of feedback received (from peers or instructors) and how it improved your approach or understanding.

## Development Roadmap
* What challenges did you face, and what strategies were used to overcome these challenges?
* What new skills or tools do you plan to learn next based on your project experience? 

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. From the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people who provided support through this project.