# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:14:13 2020

@author: shivpallavi
"""

'''
ML Health Care intership
Dataset: AHS_districtwise_2012-13.csv
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
AHS_data = pd.read_csv(r"C:\Users\shivpallavi\Desktop\MLD6June ineuron\internship\data\AHS_districtwise_2012-13.csv")
AHS_data.head()

AHS_data.info()
Total_Col = AHS_data.filter(regex = 'Total').columns
len(Total_Col)
print(Total_Col.dtype)


# rename the columns as the column names are very big
AHS_data = AHS_data.rename(columns={'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Total - Lower Limit':'CI_U5MR_Total_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Total - Upper Limit':'CI_U5MR_Total_Upper',
        'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Rural - Lower Limit':'CI_U5MR_Rural_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Rural - Upper Limit':'CI_U5MR_Rural_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Urban - Lower Limit':'CI_U5MR_Urban_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Under Five Mortality Rate (U5MR) - Urban - Upper Limit':'CI_U5MR_Urban_Upper',
                                   'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Total - Lower Limit':'CI_SRB_Total_Lower',
                                  'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Total - Upper Limit':'CI_SRB_Total_Upper',
                                  'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Rural - Lower Limit':'CI_SRB_Rural_Lower',
                                 'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Rural - Upper Limit':'CI_SRB_Rural_Upper',
                                 'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Urban - Lower Limit':'CI_SRB_Urban_Lower',
                                 'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Sex Ratio at Birth - Urban - Upper Limit':'CI_SRB_Urban_Upper'})

AHS_data = AHS_data.rename(columns={'SAMPLE PARTICULARS - Sample Units - Total':'SU_Total',
                                    'SAMPLE PARTICULARS - Sample Units - Rural':'SU_Rural',
                                    'SAMPLE PARTICULARS - Sample Units - Urban':'SU_Urban',
                                    'SAMPLE PARTICULARS - Households - Total':'Households_Total',
                                    'SAMPLE PARTICULARS - Households - Rural':'Households_Rural',
                                    'SAMPLE PARTICULARS - Households - Urban':'Households_Urban',
                                    'SAMPLE PARTICULARS - Population - Total':'Population_Total',
                                    'SAMPLE PARTICULARS - Population - Rural':'Population_Rural',
                                    'SAMPLE PARTICULARS - Population - Urban':'Population_Urban',
                                    'SAMPLE PARTICULARS - Ever Married Women (aged 15-49 years) - Total':'Ever_MW_Total',
                                    'SAMPLE PARTICULARS - Ever Married Women (aged 15-49 years) - Rural':'Ever_MW_Rural',
                                    'SAMPLE PARTICULARS - Ever Married Women (aged 15-49 years) - Urban':'Ever_MW_Urban',
                                    'SAMPLE PARTICULARS - Currently Married Women (aged 15-49 years) - Total':'Current_MW_Total',
                                    'SAMPLE PARTICULARS - Currently Married Women (aged 15-49 years) - Rural':'Current_MW_Rural',
                                    'SAMPLE PARTICULARS - Currently Married Women (aged 15-49 years) - Urban':'Current_MW_Urban'})

AHS_data = AHS_data.rename(columns={'SAMPLE PARTICULARS - Children 12-23 months - Total':'Children_Total',
                                    'SAMPLE PARTICULARS - Children 12-23 months - Rural':'Children_Rural',
                                    'SAMPLE PARTICULARS - Children 12-23 months - Urban':'Children_Urban',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - SC - Total':'Housesize_SC_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - SC - Rural':'Housesize_SC_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - SC - Urban':'Housesize_SC_Urban',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - ST - Total':'Housesize_ST_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - ST - Rural':'Housesize_ST_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - ST - Urban':'Housesize_ST_Urban',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - All - Total':'Housesize_Al1_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - All - Rural':'Housesize_Al1_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Average Household Size - All - Urban':'Housesize_Al1_Urban',
                                    'HOUSEHOLD CHARACTERISTICS - Population below age 15 years (%) - Total':'Pop_below_15_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Population below age 15 years (%) - Rural':'Pop_below_15_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Population below age 15 years (%) - Urban':'Pop_below_15_Urban',
                                    'HOUSEHOLD CHARACTERISTICS - Dependency Ratio - Total':'Dependent_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Dependency Ratio - Rural':'Dependent_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Dependency Ratio - Urban':'Dependent_Urban'})

AHS_data = AHS_data.rename(columns={'HOUSEHOLD CHARACTERISTICS - Currently Married Illiterate Women aged 15-49 years (%) - Total':'Current_MIW_Total',
                                    'HOUSEHOLD CHARACTERISTICS - Currently Married Illiterate Women aged 15-49 years (%) - Rural':'Current_MIW_Rural',
                                    'HOUSEHOLD CHARACTERISTICS - Currently Married Illiterate Women aged 15-49 years (%) - Urban':'Current_MIW_Urban',
                                    'SEX RATIO - Sex Ratio at Birth - Total':'Sex_BirthRatio_Total',
                                    'SEX RATIO - Sex Ratio at Birth - Rural':'Sex_BirthRatio_Rural',
                                    'SEX RATIO - Sex Ratio at Birth - Urban':'Sex_BirthRatio_Urban',
                                    'SEX RATIO - Sex Ratio (0- 4 years) - Total':'Sex_Ratio(0-4)_Total',
                                    'SEX RATIO - Sex Ratio (0- 4 years) - Rural':'Sex_Ratio(0-4)_Rural',
                                    'SEX RATIO - Sex Ratio (0- 4 years) - Urban':'Sex_Ratio(0-4)_Urban',
                                    'SEX RATIO - Sex Ratio (All ages) - Total':'Sex_RatioAll_Total',
                                    'SEX RATIO - Sex Ratio (All ages) - Rural':'Sex_RatioAll_Rural',
                                    'SEX RATIO - Sex Ratio (All ages) - Urban':'Sex_RatioAll_Urban',
                                    'EFFECTIVE LITERACY RATE - Person - Total':'Literacy_Total',
                                    'EFFECTIVE LITERACY RATE - Person - Rural':'Literacy_Rural',
                                    'EFFECTIVE LITERACY RATE - Person - Urban':'Literacy_Urban',
                                    'EFFECTIVE LITERACY RATE - Male - Total':'Literacy_Male_Total',
                                    'EFFECTIVE LITERACY RATE - Male - Rural':'Literacy_Male_Rural',
                                    'EFFECTIVE LITERACY RATE - Male - Urban':'Literacy_Male_Urban',
                                    'EFFECTIVE LITERACY RATE - Female - Total':'Literacy_Female_Total',
                                    'EFFECTIVE LITERACY RATE - Female - Rural':'Literacy_Female_Rural',
                                    'EFFECTIVE LITERACY RATE - Female - Urban':'Literacy_Female_Urban'})


AHS_data = AHS_data.rename(columns={'MARRIAGE - Marriages among Females below legal age (18 years) (%)# - Total':'Marriage_F<18_Total',
                                    'MARRIAGE - Marriages among Females below legal age (18 years) (%)# - Rural':'Marriage_F<18_Rural',
                                    'MARRIAGE - Marriages among Females below legal age (18 years) (%)# - Urban':'Marriage_F<18_Urban',
                                    'MARRIAGE - Marriages among Males below legal age (21 years) (%)# - Total':'Marriage_M<21_Total',
                                    'MARRIAGE - Marriages among Males below legal age (21 years) (%)# - Rural':'Marriage_M<21_Rural',
                                    'MARRIAGE - Marriages among Males below legal age (21 years) (%)# - Urban':'Marriage_M<21_Urban',
                                    'MARRIAGE - Currently Married Women aged 20-24 years married before legal age (18 years) (%) - Total':'CMW_age20-24_married<18_Total',
                                    'MARRIAGE - Currently Married Women aged 20-24 years married before legal age (18 years) (%) - Rural':'CMW_age20-24_married<18_Rural',
                                    'MARRIAGE - Currently Married Women aged 20-24 years married before legal age (18 years) (%) - Urban':'CMW_age20-24_married<18_Urban',
                                    'MARRIAGE - Currently Married Men aged 25-29 years married before legal age (21 years) (%) - Total':'CMM_age25-29_married<21_Total',
                                    'MARRIAGE - Currently Married Men aged 25-29 years married before legal age (21 years) (%) - Rural':'CMM_age25-29_married<21_Rural',
                                    'MARRIAGE - Currently Married Men aged 25-29 years married before legal age (21 years) (%) - Urban':'CMM_age25-29_married<21_Urban',
                                    'MARRIAGE - Mean age at Marriage# - Male - Total':'AvgM_MarriageAge_Total',
                                    'MARRIAGE - Mean age at Marriage# - Male - Rural':'AvgM_MarriageAge_Rural',
                                    'MARRIAGE - Mean age at Marriage# - Male - Urban':'AvgM_MarriageAge_Urban',
                                    'MARRIAGE - Mean age at Marriage# - Female - Total':'AvgF_MarriageAge_Total',
                                    'MARRIAGE - Mean age at Marriage# - Female - Rural':'AvgF_MarriageAge_Rural',
                                    'MARRIAGE - Mean age at Marriage# - Female - Urban':'AvgF_MarriageAge_Urban',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Person - Total':'SC(6-17)_Total',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Person - Rural':'SC(6-17)_Rural',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Person - Urban':'SC(6-17)_Urban',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Male - Total':'SC(6-17)_Male_Total',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Male - Rural':'SC(6-17)_Male_Rural',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Male - Urban':'SC(6-17)_Male_Urban'})


AHS_data = AHS_data.rename(columns={'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Female - Total':'SC(6-17)_Female_Total',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Female - Rural':'SC(6-17)_Female_Rural',
                                    'SCHOOLING STATUS - Children currently attending school (Age 6-17 years) (%) - Female - Urban':'SC(6-17)_Female_Urban',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Person - Total':'SCD(6-17)_Total',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Person - Rural':'SCD(6-17)_Rural',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Person - Urban':'SCD(6-17)_Urban',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Male - Total':'SCD(6-17)_Male_Total',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Male - Rural':'SCD(6-17)_Male_Rural',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Male - Urban':'SCD(6-17)_Male_Urban',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Female - Total':'SCD(6-17)_Female_Total',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Female - Rural':'SCD(6-17)_Female_Rural',
                                    'SCHOOLING STATUS - Children attended before / Drop out (Age 6-17 years) (%) - Female - Urban':'SCD(6-17)_Female_Urban',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Person - Total':'Child_Lab(5-14)_Total',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Person - Rural':'Child_Lab(5-14)_Rural',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Person - Urban':'Child_Lab(5-14)_Urban',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Male - Total':'Child_Lab(5-14)M_Total',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Male - Rural':'Child_Lab(5-14)M_Rural',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Male - Urban':'Child_Lab(5-14)M_Urban',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Female - Total':'Child_Lab(5-14)F_Total',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Female - Rural':'Child_Lab(5-14)F_Rural',
                                    'WORK STATUS - Children aged 5-14 years engaged in work (%) - Female - Urban':'Child_Lab(5-14)F_Urban'})

AHS_data = AHS_data.rename(columns={'WORK STATUS - Work Participation Rate (15 years and above) - Person - Total':'WS>=15_Total',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Person - Rural':'WS>=15_Rural',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Person - Urban':'WS>=15_Urban',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Male - Total':'WS>=15M_Total',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Male - Rural':'WS>=15M_Rural',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Male - Urban':'WS>=15M_Urban',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Female - Total':'WS>=15F_Total',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Female - Rural':'WS>=15F_Rural',
                                    'WORK STATUS - Work Participation Rate (15 years and above) - Female - Urban':'WS>=15F_Urban',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Person - Total':'Disable_Total',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Person - Rural':'Disable_Rural',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Person - Urban':'Disable_Urban',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Male - Total':'Disable_Male_Total',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Male - Rural':'Disable_Male_Rural',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Male - Urban':'Disable_Male_Urban',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Female - Total':'Disable_Female_Total',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Female - Rural':'Disable_Female_Rural',
                                    'DISABILITY - Prevalence of any type of Disability (Per 100,000 Population) - Female - Urban':'Disable_Female_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Person - Total':'Inj_Treated_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Person - Rural':'Inj_Treated_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Person - Urban':'Inj_Treated_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Male - Total':'Inj_Treated_M_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Male - Rural':'Inj_Treated_M_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Male - Urban':'Inj_Treated_M_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Female - Total':'Inj_Treated_F_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Female - Rural':'Inj_Treated_F_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Severe - Female - Urban':'Inj_Treated_F_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Person - Total':'MaInj_Treated_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Person - Rural':'MaInj_Treated_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Person - Urban':'MaInj_Treated_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Male - Total':'MaInj_Treated_M_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Male - Rural':'MaInj_Treated_M_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Male - Urban':'MaInj_Treated_M_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Female - Total':'MaInj_Treated_F_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Female - Rural':'MaInj_Treated_F_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Major - Female - Urban':'MaInj_Treated_F_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Person - Total':'MiInj_Treated_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Person - Rural':'MiInj_Treated_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Person - Urban':'MiInj_Treated_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Male - Total':'MiInj_Treated_M_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Male - Rural':'MiInj_Treated_M_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Male - Urban':'MiInj_Treated_M_Urban',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Female - Total':'MiInj_Treated_F_Total',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Female - Rural':'MiInj_Treated_F_Rural',
                                    'INJURY - Number of Injured Persons by type of Treatment received (Per 100,000 Population) - Minor - Female - Urban':'MiInj_Treated_F_Urban'})


AHS_data = AHS_data.rename(columns={'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Person - Total':'Diarrhoea_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Person - Rural':'Diarrhoea_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Person - Urban':'Diarrhoea_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Male - Total':'Diarrhoea_M_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Male - Rural':'Diarrhoea_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Male - Urban':'Diarrhoea_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Female - Total':'Diarrhoea_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Female - Rural':'Diarrhoea_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Diarrhoea/Dysentery - Female - Urban':'Diarrhoea_F_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Person - Total':'ARI_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Person - Rural':'ARI_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Person - Urban':'ARI_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Male - Total':'ARI_M_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Male - Rural':'ARI_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Male - Urban':'ARI_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Female - Total':'ARI_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Female - Rural':'ARI_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Acute Respiratory Infection (ARI) - Female - Urban':'ARI_F_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Person - Total':'Fever_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Person - Rural':'Fever_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Person - Urban':'Fever_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Male - Total':'Fever_M_total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Male - Rural':'Fever_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Male - Urban':'Fever_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Female - Total':'Fever_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Female - Rural':'Fever_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Fever (All Types) - Female - Urban':'Fever_F_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Person - Total':'AnyIllness_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Person - Rural':'AnyIllness_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Person - Urban':'AntIllness_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Male - Total':'AnyIllness_M_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Male - Rural':'AnyIllness_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Male - Urban':'AnyIllness_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Female - Total':'AnyIllness_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Female - Rural':'AnyIllness_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness (Per 100,000 Population) - Any type of Acute Illness - Female - Urban':'AntIllness_F_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Person - Total':'Any_Treatment_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Person - Rural':'Any_Treatment_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Person - Urban':'Any_Treatment_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Male - Total':'Any_Treatment_M_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Male - Rural':'Any_Treatment_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Male - Urban':'Any_Treatment_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Female - Total':'Any_Treatment_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Female - Rural':'Any_Treatment_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Any Source (%) - Female - Urban':'Any_Treatment_F_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Person - Total':'Gov_Treatment_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Person - Rural':'Gov_Treatment_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Person - Urban':'Gov_Treatment_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Male - Total':'Gov_Treatment_M_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Male - Rural':'Gov_Treatment_M_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Male - Urban':'Gov_Treatment_M_Urban',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Female - Total':'Gov_Treatment_F_Total',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Female - Rural':'Gov_Treatment_F_Rural',
                                    'ACUTE ILLNESS - Persons suffering from Acute Illness and taking treatment from Government Source (%) - Female - Urban':'Gov_Treatment_F_Urban'})


AHS_data = AHS_data.rename(columns={'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Person - Total':'CI_AnySym_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Person - Rural':'CI_AnySym_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Person - Urban':'CI_AnySym_Urban',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Male - Total':'CI_AnySym_M_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Male - Rural':'CI_AnySym_M_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Male - Urban':'CI_AnySym_M_Urban',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Female - Total':'CI_AnySym_F_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Female - Rural':'CI_AnySym_F_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness (Per 100,000 Population) - Female - Urban':'CI_AnySym_F_Urban',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Person - Total':'CI_SMed_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Person - Rural':'CI_SMed_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Person - Urban':'CI_SMed_Urban',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Male - Total':'CI_SMed_M_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Male - Rural':'CI_SMed_M_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Male - Urban':'CI_SMed_M_Urban',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Female - Total':'CI_SMed_F_Total',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Female - Rural':'CI_SMed_F_Rural',
                                    'CHRONIC ILLNESS - Having any kind of Symptoms of Chronic Illness and sought Medical Care (%) - Female - Urban':'CI_SMed_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Person - Total':'CI_Diabetes_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Person - Rural':'CI_Diabetes_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Person - Urban':'CI_Diabetes_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Male - Total':'CI_Diabetes_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Male - Rural':'CI_Diabetes_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Male - Urban':'CI_Diabetes_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Female - Total':'CI_Diabetes_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Female - Rural':'CI_Diabetes_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Diabetes - Female - Urban':'CI_Diabetes_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Person - Total':'CI_HighBP_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Person - Rural':'CI_HighBP_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Person - Urban':'CI_HighBP_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Male - Total':'CI_HighBP_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Male - Rural':'CI_HighBP_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Male - Urban':'CI_HighBP_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Female - Total':'CI_HighBP_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Female - Rural':'CI_HighBP_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Hypertension - Female - Urban':'CI_HighBP_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Person - Total':'CI_TB_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Person - Rural':'CI_TB_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Person - Urban':'CI_TB_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Male - Total':'CI_TB_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Male - Rural':'CI_TB_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Male - Urban':'CI_TB_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Female - Total':'CI_TB_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Female - Rural':'CI_TB_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Tuberculosis (TB) - Female - Urban':'CI_TB_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Person - Total':'CI_Asthma_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Person - Rural':'CI_Asthma_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Person - Urban':'CI_Asthma_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Male - Total':'CI_Asthma_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Male - Rural':'CI_Asthma_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Male - Urban':'CI_Asthma_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Female - Total':'CI_Asthma_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Female - Rural':'CI_Asthma_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Asthma / Chronic Respiratory Disease - Female - Urban':'CI_Asthma_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Person - Total':'CI_Arthritis_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Person - Rural':'CI_Arthritis_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Person - Urban':'CI_Arthritis_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Male - Total':'CI_Arthritis_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Male - Rural':'CI_Arthritis_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Male - Urban':'CI_Arthritis_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Female - Total':'CI_Arthritis_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Female - Rural':'CI_Arthritis_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Arthritis - Female - Urban':'CI_Arthritis_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Person - Total':'CI_AnyKind_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Person - Rural':'CI_AnyKind_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Person - Urban':'CI_AnyKind_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Male - Total':'CI_AnyKind_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Male - Rural':'CI_AnyKind_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Male - Urban':'CI_AnyKind_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Female - Total':'CI_AnyKind_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Female - Rural':'CI_AnyKind_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for Chronic Illness (Per 100,000 Population) - Any kind of Chronic Illness - Female - Urban':'CI_AnyKind_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Person - Total':'CI_RT_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Person - Rural':'CI_RT_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Person - Urban':'CI_RT_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Male - Total':'CI_RT_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Male - Rural':'CI_RT_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Male - Urban':'CI_RT_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Female - Total':'CI_RT_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Female - Rural':'CI_RT_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment (%) - Female - Urban':'CI_RT_F_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Person - Total':'CI_RTGov_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Person - Rural':'CI_RTGov_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Person - Urban':'CI_RTGov_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Male - Total':'CI_RTGov_M_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Male - Rural':'CI_RTGov_M_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Male - Urban':'CI_RTGov_M_Urban',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Female - Total':'CI_RTGov_F_Total',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Female - Rural':'CI_RTGov_F_Rural',
                                    'CHRONIC ILLNESS - Having diagnosed for any kind of Chronic Illness and getting Regular Treatment from Government Source (%) - Female - Urban':'CI_RTGov_F_Urban'})


AHS_data = AHS_data.rename(columns={'FERTILITY - Crude Birth Rate (CBR) - Total':'Fertility_CBR_Total',
                                    'FERTILITY - Crude Birth Rate (CBR) - Rural':'Fertility_CBR_Rural',
                                    'FERTILITY - Crude Birth Rate (CBR) - Urban':'Fertility_CBR_Urban',
                                    'FERTILITY - Natural Growth Rate - Total':'Fertility_NGR_Total',
                                    'FERTILITY - Natural Growth Rate - Rural':'Fertility_NGR_Rural',
                                    'FERTILITY - Natural Growth Rate - Urban':'Fertility_NGR_Urban',
                                    'FERTILITY - Total Fertility Rate - Total':'Fertility_TFR_Total',
                                    'FERTILITY - Total Fertility Rate - Rural':'Fertility_TFR_Rural',
                                    'FERTILITY - Total Fertility Rate - Urban':'Fertility_TFR_Urban',
                                    'FERTILITY - Women aged 20-24 reporting birth of order 2 & above (%) - Total':'W(20-24)_RepO>=2_Total',
                                    'FERTILITY - Women aged 20-24 reporting birth of order 2 & above (%) - Rural':'W(20-24)_RepO>=2_Rural',
                                    'FERTILITY - Women aged 20-24 reporting birth of order 2 & above (%) - Urban':'W(20-24)_RepO>=2_Urban',
                                    'FERTILITY - Women reporting birth of order 3 & above (%) - Total':'W_RepO>=3_Total',
                                    'FERTILITY - Women reporting birth of order 3 & above (%) - Rural':'W_RepO>=3_Rural',
                                    'FERTILITY - Women reporting birth of order 3 & above (%) - Urban':'W_RepO>=3_Urban',
                                    'FERTILITY - Women with two children wanting no more children (%) - Total':'W_2nNoMoreCh_Total',
                                    'FERTILITY - Women with two children wanting no more children (%) - Rural':'W_2nNoMoreCh_Rural',
                                    'FERTILITY - Women with two children wanting no more children (%) - Urban':'W_2nNoMoreCh_Urban',
                                    'FERTILITY - Women aged 15-19 years who were already mothers or pregnant at the time of survey (%) - Total':'W(15-19)_Pre_Total',
                                    'FERTILITY - Women aged 15-19 years who were already mothers or pregnant at the time of survey (%) - Rural':'W(15-19)_Pre_Rural',
                                    'FERTILITY - Women aged 15-19 years who were already mothers or pregnant at the time of survey (%) - Urban':'W(15-19)_Pre_Urban',
                                    'FERTILITY - Median age at first live birth of Women aged 15-49 years - Total':'Med_W(15-19)_LB_Total',
                                    'FERTILITY - Median age at first live birth of Women aged 15-49 years - Rural':'Med_W(15-19)_LB_Rural',
                                    'FERTILITY - Median age at first live birth of Women aged 15-49 years - Urban':'Med_W(15-19)_LB_Urban',
                                    'FERTILITY - Median age at first live birth of Women aged 25-49 years - Total':'Med_W(25-49)_LB_Total',
                                    'FERTILITY - Median age at first live birth of Women aged 25-49 years - Rural':'Med_W(25-49)_LB_Rural',
                                    'FERTILITY - Median age at first live birth of Women aged 25-49 years - Urban':'Med_W(25-49)_LB_Urban',
                                    'FERTILITY - Live Births taking place after an interval of 36 months (%) - Total':'LB_Aft36m_Total',
                                    'FERTILITY - Live Births taking place after an interval of 36 months (%) - Rural':'LB_Aft36m_Rural',
                                    'FERTILITY - Live Births taking place after an interval of 36 months (%) - Urban':'LB_Aft36m_Urban',
                                    'FERTILITY - Mean number of children ever born to Women aged 15-49 years - Total':'MNoCh_W(15-49)_Total',
                                    'FERTILITY - Mean number of children ever born to Women aged 15-49 years - Rural':'MNoCh_W(15-49)_Rural',
                                    'FERTILITY - Mean number of children ever born to Women aged 15-49 years - Urban':'MNoCh_W(15-49)_Urban',
                                    'FERTILITY - Mean number of children surviving to Women aged 15-49 years - Total':'MNoChSur_W(15-49)_Total',
                                    'FERTILITY - Mean number of children surviving to Women aged 15-49 years - Rural':'MNoChSur_W(15-49)_Rural',
                                    'FERTILITY - Mean number of children surviving to Women aged 15-49 years - Urban':'MNoChSur_W(15-49)_Urban',
                                    'FERTILITY - Mean number of children ever born to Women aged 45-49 years - Total':'MNoCh_W(45-49)_Total',
                                    'FERTILITY - Mean number of children ever born to Women aged 45-49 years - Rural':'MNoCh_W(45-49)_Rural',
                                    'FERTILITY - Mean number of children ever born to Women aged 45-49 years - Urban':'MNoCh_W(45-49)_Urban'})


AHS_data = AHS_data.rename(columns={'ABORTION - Pregnancy to Women aged 15-49 years resulting in abortion (%) - Total':'Abort_W(15-49)_Total',
                                    'ABORTION - Pregnancy to Women aged 15-49 years resulting in abortion (%) - Rural':'Abort_W(15-49)_Rural',
                                    'ABORTION - Pregnancy to Women aged 15-49 years resulting in abortion (%) - Urban':'Abort_W(15-49)_Urban',
                                    'ABORTION - Women who received any ANC before abortion (%) - Total':'Abort_ANCRec_Total',
                                    'ABORTION - Women who received any ANC before abortion (%) - Rural':'Abort_ANCRec_Rural',
                                    'ABORTION - Women who received any ANC before abortion (%) - Urban':'Abort_ANCRec_Urban',
                                    'ABORTION - Women who went for Ultrasound before abortion (%) - Total':'Abort_USDOne_Total',
                                    'ABORTION - Women who went for Ultrasound before abortion (%) - Rural':'Abort_USDOne_Rural',
                                    'ABORTION - Women who went for Ultrasound before abortion (%) - Urban':'Abort_USDOne_Urban',
                                    'ABORTION - Average Month of pregnancy at the time of abortion - Total':'Abort_AVgMth_Total',
                                    'ABORTION - Average Month of pregnancy at the time of abortion - Rural':'Abort_AVgMth_Rural',
                                    'ABORTION - Average Month of pregnancy at the time of abortion - Urban':'Abort_AVgMth_Urban',
                                    'ABORTION - Abortion performed by skilled health personnel (%) - Total':'Abort_SHP_Total',
                                    'ABORTION - Abortion performed by skilled health personnel (%) - Rural':'Abort_SHP_Rural',
                                    'ABORTION - Abortion performed by skilled health personnel (%) - Urban':'Abort_SHP_Urban',
                                    'ABORTION - Abortion taking place in Institution (%) - Total':'Abort_Inst_Total',
                                    'ABORTION - Abortion taking place in Institution (%) - Rural':'Abort_Inst_Rural',
                                    'ABORTION - Abortion taking place in Institution (%) - Urban':'Abort_Inst_Urban'
                                    })

AHS_data = AHS_data.rename(columns={'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Method (%) - Total':'FP_AnyMethod_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Method (%) - Rural':'FP_AnyMethod_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Method (%) - Urban':'FP_AnyMethod_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Modern Method (%) - Total':'FP_AnyMM_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Modern Method (%) - Rural':'FP_AnyMM_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Modern Method (%) - Urban':'FP_AnyMM_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Female Sterilization (%) - Total':'FP_FSter_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Female Sterilization (%) - Rural':'FP_FSter_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Female Sterilization (%) - Urban':'FP_FSter_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Male Sterilization (%) - Total':'FP_MSter_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Male Sterilization (%) - Rural':'FP_MSter_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Male Sterilization (%) - Urban':'FP_MSter_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Copper-T/IUD (%) - Total':'FP_IUD_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Copper-T/IUD (%) - Rural':'FP_IUD_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Copper-T/IUD (%) - Urban':'FP_IUD_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Pills (%) - Total':'FP_Pills_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Pills (%) - Rural':'FP_Pills_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Pills (%) - Urban':'FP_Pills_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Condom/Nirodh (%) - Total':'FP_Condom_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Condom/Nirodh (%) - Rural':'FP_Condom_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Condom/Nirodh (%) - Urban':'FP_Condom_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Emergency Contraceptive Pills (%) - Total':'FP_ECPills_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Emergency Contraceptive Pills (%) - Rural':'FP_ECPills_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Emergency Contraceptive Pills (%) - Urban':'FP_ECPills_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Traditional Method (%) - Total':'FP_ATM_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Traditional Method (%) - Rural':'FP_ATM_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Any Traditional Method (%) - Urban':'FP_ATM_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Periodic Abstinence (%) - Total':'FP_PA_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Periodic Abstinence (%) - Rural':'FP_PA_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Periodic Abstinence (%) - Urban':'FP_PA_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Withdrawal (%) - Total':'FP_Withdrawal_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Withdrawal (%) - Rural':'FP_Withdrawal_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - Withdrawal (%) - Urban':'FP_Withdrawal_Urban',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - LAM (%) - Total':'FP_LAM_Total',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - LAM (%) - Rural':'FP_LAM_Rural',
                                    'FAMILY PLANNING PRACTICES (CMW AGED 15-49 YEARS) - Current Usage - LAM (%) - Urban':'FP_LAM_Urban',
                                    })


AHS_data = AHS_data.rename(columns={'UNMET NEED FOR FAMILY PLANNING - Unmet need for Spacing (%) - Total':'UNFP_Space_Total',
                                    'UNMET NEED FOR FAMILY PLANNING - Unmet need for Spacing (%) - Rural':'UNFP_Space_Rural',
                                    'UNMET NEED FOR FAMILY PLANNING - Unmet need for Spacing (%) - Urban':'UNFP_Space_Urban',
                                    'UNMET NEED FOR FAMILY PLANNING - Unmet need for Limiting (%) - Total':'UNFP_Limit_Total',
                                    'UNMET NEED FOR FAMILY PLANNING - Unmet need for Limiting (%) - Rural':'UNFP_Limit_Rural',
                                    'UNMET NEED FOR FAMILY PLANNING - Unmet need for Limiting (%) - Urban':'UNFP_Limit_Urban',
                                    'UNMET NEED FOR FAMILY PLANNING - Total Unmet need (%) - Total':'UNFP_per_Total',
                                    'UNMET NEED FOR FAMILY PLANNING - Total Unmet need (%) - Rural':'UNFP_per_Rural',
                                    'UNMET NEED FOR FAMILY PLANNING - Total Unmet need (%) - Urban':'UNFP_per_Urban'})

AHS_data = AHS_data.rename(columns={'ANTE NATAL CARE - Currently Married Pregnant Women aged 15-49 years registered for ANC (%) - Total':'ANC_Reg_Total',
                                    'ANTE NATAL CARE - Currently Married Pregnant Women aged 15-49 years registered for ANC (%) - Rural':'ANC_Reg_Rural',
                                    'ANTE NATAL CARE - Currently Married Pregnant Women aged 15-49 years registered for ANC (%) - Urban':'ANC_Reg_Urban',
                                    'ANTE NATAL CARE - Mothers who received any Antenatal Check-up (%) - Total':'ANC_Rec_Total',
                                    'ANTE NATAL CARE - Mothers who received any Antenatal Check-up (%) - Rural':'ANC_Rec_Rural',
                                    'ANTE NATAL CARE - Mothers who received any Antenatal Check-up (%) - Urban':'ANC_Rec_Urban',
                                    'ANTE NATAL CARE - Mothers who had Antenatal Check-up in First Trimester (%) - Total':'ANC_1Tri_Total',
                                    'ANTE NATAL CARE - Mothers who had Antenatal Check-up in First Trimester (%) - Rural':'ANC_1Tri_Rural',
                                    'ANTE NATAL CARE - Mothers who had Antenatal Check-up in First Trimester (%) - Urban':'ANC_1Tri_Urban',
                                    'ANTE NATAL CARE - Mothers who received 3 or more Antenatal Care (%) - Total':'ANC_Rec>=3_Total',
                                    'ANTE NATAL CARE - Mothers who received 3 or more Antenatal Care (%) - Rural':'ANC_Rec>=3_Rural',
                                    'ANTE NATAL CARE - Mothers who received 3 or more Antenatal Care (%) - Urban':'ANC_Rec>=3_Urban',
                                    'ANTE NATAL CARE - Mothers who received at least one Tetanus Toxoid (TT) injection (%) - Total':'ANC_RecTT_Total',
                                    'ANTE NATAL CARE - Mothers who received at least one Tetanus Toxoid (TT) injection (%) - Rural':'ANC_RecTT_Rural',
                                    'ANTE NATAL CARE - Mothers who received at least one Tetanus Toxoid (TT) injection (%) - Urban':'ANC_RecTT_Urban',
                                    'ANTE NATAL CARE - Mothers who consumed IFA for 100 days or more (%) - Total':'ANC_ConsIFA>=100d_Total',
                                    'ANTE NATAL CARE - Mothers who consumed IFA for 100 days or more (%) - Rural':'ANC_ConsIFA>=100d_Rural',
                                    'ANTE NATAL CARE - Mothers who consumed IFA for 100 days or more (%) - Urban':'ANC_ConsIFA>=100d_Urban',
                                    'ANTE NATAL CARE - Mothers who had Full Antenatal Check-up (%) - Total':'ANC_FullCheckup_Total',
                                    'ANTE NATAL CARE - Mothers who had Full Antenatal Check-up (%) - Rural':'ANC_FullCheckup_Rural',
                                    'ANTE NATAL CARE - Mothers who had Full Antenatal Check-up (%) - Urban':'ANC_FullCheckup_Urban',
                                    'ANTE NATAL CARE - Mothers who received ANC from Govt. Source (%) - Total':'ANC_Gov_Total',
                                    'ANTE NATAL CARE - Mothers who received ANC from Govt. Source (%) - Rural':'ANC_Gov_Rural',
                                    'ANTE NATAL CARE - Mothers who received ANC from Govt. Source (%) - Urban':'ANC_Gov_Urban',
                                    'ANTE NATAL CARE - Mothers whose Blood Pressure (BP) taken (%) - Total':'ANC_BPtaken_Total',
                                    'ANTE NATAL CARE - Mothers whose Blood Pressure (BP) taken (%) - Rural':'ANC_BPtaken_Rural',
                                    'ANTE NATAL CARE - Mothers whose Blood Pressure (BP) taken (%) - Urban':'ANC_BPtaken_Urban',
                                    'ANTE NATAL CARE - Mothers whose Blood taken for Hb (%) - Total':'ANC_Hbtaken_Total',
                                    'ANTE NATAL CARE - Mothers whose Blood taken for Hb (%) - Rural':'ANC_Hbtaken_Rural',
                                    'ANTE NATAL CARE - Mothers whose Blood taken for Hb (%) - Urban':'ANC_Hbtaken_Urban',
                                    'ANTE NATAL CARE - Mothers who underwent Ultrasound (%) - Total':'ANC_US_Total',
                                    'ANTE NATAL CARE - Mothers who underwent Ultrasound (%) - Rural':'ANC_US_Rural',
                                    'ANTE NATAL CARE - Mothers who underwent Ultrasound (%) - Urban':'ANC_US_Urban'
                                    })

AHS_data = AHS_data.rename(columns={'DELIVERY CARE - Institutional Delivery (%) - Total':'DC_InstDel_Total',
                                    'DELIVERY CARE - Institutional Delivery (%) - Rural':'DC_InstDel_Rural',
                                    'DELIVERY CARE - Institutional Delivery (%) - Urban':'DC_InstDel_Urban',
                                    'DELIVERY CARE - Delivery at Government Institution (%) - Total':'DC_GovInstDel_Total',
                                    'DELIVERY CARE - Delivery at Government Institution (%) - Rural':'DC_GovInstDel_Rural',
                                    'DELIVERY CARE - Delivery at Government Institution (%) - Urban':'DC_GovInstDel_Urban',
                                    'DELIVERY CARE - Delivery at Private Institution (%) - Total':'DC_PrivInstDel_Total',
                                    'DELIVERY CARE - Delivery at Private Institution (%) - Rural':'DC_PrivInstDel_Rural',
                                    'DELIVERY CARE - Delivery at Private Institution (%) - Urban':'DC_PrivInstDel_Urban',
                                    'DELIVERY CARE - Delivery at Home (%) - Total':'DC_Home_Total',
                                    'DELIVERY CARE - Delivery at Home (%) - Rural':'DC_Home_Rural',
                                    'DELIVERY CARE - Delivery at Home (%) - Urban':'DC_Home_Urban',
                                    'DELIVERY CARE - Delivery at home conducted by skilled health personnel (%) - Total':'DC_HomeSHP_Total',
                                    'DELIVERY CARE - Delivery at home conducted by skilled health personnel (%) - Rural':'DC_HomeSHP_Rural',
                                    'DELIVERY CARE - Delivery at home conducted by skilled health personnel (%) - Urban':'DC_HomeSHP_Urban',
                                    'DELIVERY CARE - Safe Delivery (%) - Total':'DC_SD_Total',
                                    'DELIVERY CARE - Safe Delivery (%) - Rural':'DC_SD_Rural',
                                    'DELIVERY CARE - Safe Delivery (%) - Urban':'DC_SD_Urban',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Government Institutions (%) - Total':'DC_GovC-Section_Total',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Government Institutions (%) - Rural':'DC_GovC-Section_Rural',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Government Institutions (%) - Urban':'DC_GovC-Section_Urban',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Private Institutions (%) - Total':'DC_PrivC-Section_Total',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Private Institutions (%) - Rural':'DC_PrivC-Section_Rural',
                                    'DELIVERY CARE - Caesarean out of total delivery taken place in Private Institutions (%) - Urban':'DC_PrivC-Section_Urban',
                                    })


AHS_data = AHS_data.rename(columns={'POST NATAL CARE - Less than 24 hrs. stay in institution after delivery (%) - Total':'PNC_aftDel<24hrs_Total',
                                    'POST NATAL CARE - Less than 24 hrs. stay in institution after delivery (%) - Rural':'PNC_aftDel<24hrs_Rural',
                                    'POST NATAL CARE - Less than 24 hrs. stay in institution after delivery (%) - Urban':'PNC_aftDel<24hrs_Urban',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 48 hrs. of delivery (%) - Total':'PNC_check48hrs_Total',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 48 hrs. of delivery (%) - Rural':'PNC_check48hrs_Rural',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 48 hrs. of delivery (%) - Urban':'PNC_check48hrs_Urban',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 1 week of delivery (%) - Total':'PNC_1week_Total',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 1 week of delivery (%) - Rural':'PNC_1week_Rural',
                                    'POST NATAL CARE - Mothers who received Post-natal Check-up within 1 week of delivery (%) - Urban':'PNC_1week_Urban',
                                    'POST NATAL CARE - Mothers who did not receive any Post-natal Check-up (%) - Total':'PNC_No_Total',
                                    'POST NATAL CARE - Mothers who did not receive any Post-natal Check-up (%) - Rural':'PNC_No_Rural',
                                    'POST NATAL CARE - Mothers who did not receive any Post-natal Check-up (%) - Urban':'PNC_No_Urban',
                                    'POST NATAL CARE - New borns who were checked up within 24 hrs. of birth (%) - Total':'PNC_NewBornCheck24_Total',
                                    'POST NATAL CARE - New borns who were checked up within 24 hrs. of birth (%) - Rural':'PNC_NewBornCheck24_Rural',
                                    'POST NATAL CARE - New borns who were checked up within 24 hrs. of birth (%) - Urban':'PNC_NewBornCheck24_Urban',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for delivery under JSY (%) - Total':'JSY_Total',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for delivery under JSY (%) - Rural':'JSY_Rural',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for delivery under JSY (%) - Urban':'JSY_Urban',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for institutional delivery under JSY (%) - Total':'JSY_InstDel_Total',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for institutional delivery under JSY (%) - Rural':'JSY_InstDel_Rural',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for institutional delivery under JSY (%) - Urban':'JSY_InstDel_Urban',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for Government Institutional delivery under JSY (%) - Total':'JSY_GovInstDel_Total',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for Government Institutional delivery under JSY (%) - Rural':'JSY_GovInstDel_Rural',
                                    'JANANI SURAKSHA YOJANA (JSY) - Mothers who availed financial assistance for Government Institutional delivery under JSY (%) - Urban':'JSY_GovInstDel_Urban'})



AHS_data = AHS_data.rename(columns={'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months having Immunization Card (%) - Total':'IMM_HaveCard_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months having Immunization Card (%) - Rural':'IMM_HaveCard_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months having Immunization Card (%) - Urban':'IMM_HaveCard_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received BCG (%) - Total':'IMM_recBCG_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received BCG (%) - Rural':'IMM_recBCG_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received BCG (%) - Urban':'IMM_recBCG_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of Polio vaccine (%) - Total':'IMM_recPolioVac_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of Polio vaccine (%) - Rural':'IMM_recPolioVac_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of Polio vaccine (%) - Urban':'IMM_recPolioVac_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of DPT vaccine (%) - Total':'IMM_recDPT_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of DPT vaccine (%) - Rural':'IMM_recDPT_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received 3 doses of DPT vaccine (%) - Urban':'IMM_recDPT_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received Measles vaccine (%) - Total':'IMM_recMeasles_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received Measles vaccine (%) - Rural':'IMM_recMeasles_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months who have received Measles vaccine (%) - Urban':'IMM_recMeasles_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months Fully Immunized (%) - Total':'IMM_FullImm_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months Fully Immunized (%) - Rural':'IMM_FullImm_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children aged 12-23 months Fully Immunized (%) - Urban':'IMM_FullImm_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who have received Polio dose at birth (%) - Total':'IMM_BirthPolioVac_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who have received Polio dose at birth (%) - Rural':'IMM_BirthPolioVac_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who have received Polio dose at birth (%) - Urban':'IMM_BirthPolioVac_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who did not receive any vaccination (%) - Total':'IMM_NoVac_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who did not receive any vaccination (%) - Rural':'IMM_NoVac_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children who did not receive any vaccination (%) - Urban':'IMM_NoVac_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received at least one Vitamin A dose during last six months (%) - Total':'IMM_recVitA_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received at least one Vitamin A dose during last six months (%) - Rural':'IMM_recVitA_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received at least one Vitamin A dose during last six months (%) - Urban':'IMM_recVitA_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received IFA tablets/syrup during last 3 months (%) - Total':'IMM_recIFA3mths_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received IFA tablets/syrup during last 3 months (%) - Rural':'IMM_recIFA3mths_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children (aged 6-35 months) who received IFA tablets/syrup during last 3 months (%) - Urban':'IMM_recIFA3mths_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children whose birth weight was taken (%) - Total':'IMM_BirthWeightTaken_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children whose birth weight was taken (%) - Rural':'IMM_BirthWeightTaken_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children whose birth weight was taken (%) - Urban':'IMM_BirthWeightTaken_Urban',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children with birth weight less than 2.5 Kg. (%) - Total':'IMM_BirthWeight<2.5_Total',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children with birth weight less than 2.5 Kg. (%) - Rural':'IMM_BirthWeight<2.5_Rural',
                                    'IMMUNIZATION, VITAMIN A & IRON SUPPLEMENT AND BIRTH WEIGHT - Children with birth weight less than 2.5 Kg. (%) - Urban':'IMM_BirthWeight<2.5_Urban'})


AHS_data = AHS_data.rename(columns={'CHILDHOOD DISEASES - Children suffering from Diarrhoea (%) - Total':'CD_Diarrhoea_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Diarrhoea (%) - Rural':'CD_Diarrhoea_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Diarrhoea (%) - Urban':'CD_Diarrhoea_Urban',
                                    'CHILDHOOD DISEASES - Children suffering from Diarrhoea who received HAF/ORS/ORT (%) - Total':'CD_DiaRecORS_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Diarrhoea who received HAF/ORS/ORT (%) - Rural':'CD_DiaRecORS_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Diarrhoea who received HAF/ORS/ORT (%) - Urban':'CD_DiaRecORS_Urban',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection (%) - Total':'CD_ARI_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection (%) - Rural':'CD_ARI_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection (%) - Urban':'CD_ARI_Urban',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection who sought treatment (%) - Total':'CD_ARITreated_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection who sought treatment (%) - Rural':'CD_ARITreated_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Acute Respiratory Infection who sought treatment (%) - Urban':'CD_ARITreated_Urban',
                                    'CHILDHOOD DISEASES - Children suffering from Fever (%) - Total':'CD_Fever_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Fever (%) - Rural':'CD_Fever_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Fever (%) - Urban':'CD_Fever_Urban',
                                    'CHILDHOOD DISEASES - Children suffering from Fever who sought treatment (%) - Total':'CD_FeverTreated_Total',
                                    'CHILDHOOD DISEASES - Children suffering from Fever who sought treatment (%) - Rural':'CD_FeverTreated_Rural',
                                    'CHILDHOOD DISEASES - Children suffering from Fever who sought treatment (%) - Urban':'CD_FeverTreated_Urban'})

AHS_data = AHS_data.rename(columns={'BREASTFEEDING AND SUPPLEMENTATION - Children breastfed within one hour of birth (%) - Total':'BnS_bfwithin1hr_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - Children breastfed within one hour of birth (%) - Rural':'BnS_bfwithin1hr_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - Children breastfed within one hour of birth (%) - Urban':'BnS_bfwithin1hr_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - Children (aged 6-35 months) exclusively breastfed for at least six months (%) - Total':'BnS_bffor6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - Children (aged 6-35 months) exclusively breastfed for at least six months (%) - Rural':'BnS_bffor6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - Children (aged 6-35 months) exclusively breastfed for at least six months (%) - Urban':'BnS_bffor6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Water (%) - Total':'BnS_bfnWaterAft6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Water (%) - Rural':'BnS_bfnWaterAft6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Water (%) - Urban':'BnS_bfnWaterAft6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Animal/Formula Milk (%) - Total':'BnS_bfnMilkAft6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Animal/Formula Milk (%) - Rural':'BnS_bfnMilkAft6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Animal/Formula Milk (%) - Urban':'BnS_bfnMilkAft6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Semi-Solid mashed food (%) - Total':'BnS_bfnSSfoodAft6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Semi-Solid mashed food (%) - Rural':'BnS_bfnSSfoodAft6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Semi-Solid mashed food (%) - Urban':'BnS_bfnSSfoodAft6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Solid (Adult) Food (%) - Total':'BnS_bfnSfoodAft6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Solid (Adult) Food (%) - Rural':'BnS_bfnSfoodAft6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Solid (Adult) Food (%) - Urban':'BnS_bfnSfoodAft6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Vegetables/Fruits (%) - Total':'BnS_bfnVegAft6mths_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Vegetables/Fruits (%) - Rural':'BnS_bfnVegAft6mths_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - CHILDREN WHO RECEIVED FOODS OTHER THAN BREAST MILK DURING FIRST 6 MONTHS - Vegetables/Fruits (%) - Urban':'BnS_bfnVegAft6mths_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Water - Total':'BnS_AvgMthsRecWater_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Water - Rural':'BnS_AvgMthsRecWater_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Water - Urban':'BnS_AvgMthsRecWater_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Animal/Formula Milk - Total':'BnS_AvgMthsRecMilk_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Animal/Formula Milk - Rural':'BnS_AvgMthsRecMilk_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Animal/Formula Milk - Urban':'BnS_AvgMthsRecMilk_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Semi-Solid mashed food - Total':'BnS_AvgMthsRecSS_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Semi-Solid mashed food - Rural':'BnS_AvgMthsRecSS_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Semi-Solid mashed food - Urban':'BnS_AvgMthsRecSS_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Solid (Adult) Food - Total':'BnS_AvgMthsRecS_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Solid (Adult) Food - Rural':'BnS_AvgMthsRecS_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Solid (Adult) Food - Urban':'BnS_AvgMthsRecS_Urban',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Vegetables/Fruits - Total':'BnS_AvgMthsRecVeg_Total',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Vegetables/Fruits - Rural':'BnS_AvgMthsRecVeg_Rural',
                                    'BREASTFEEDING AND SUPPLEMENTATION - AVERAGE MONTH BY WHICH CHILDREN RECEIVED FOODS OTHER THAN BREAST MILK - Vegetables/Fruits - Urban':'BnS_AvgMthsRecVeg_Urban',
                                    'BIRTH REGISTRATION - Birth Registered (%) - Total':'BR_Total',
                                    'BIRTH REGISTRATION - Birth Registered (%) - Rural':'BR_Rural',
                                    'BIRTH REGISTRATION - Birth Registered (%) - Urban':'BR_Urban',
                                    'BIRTH REGISTRATION - Children whose birth was registered and received Birth Certificate (%) - Total':'BR_RecBC_Total',
                                    'BIRTH REGISTRATION - Children whose birth was registered and received Birth Certificate (%) - Rural':'BR_RecBC_Rural',
                                    'BIRTH REGISTRATION - Children whose birth was registered and received Birth Certificate (%) - Urban':'BR_RecBC_Urban'})


AHS_data = AHS_data.rename(columns={'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HIV/AIDS (%) - Total':'Aware_W_HIV_Total',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HIV/AIDS (%) - Rural':'Aware_W_HIV_Rural',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HIV/AIDS (%) - Urban':'Aware_W_HIV_Urban',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of RTI/STI (%) - Total':'Aware_W_STI_Total',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of RTI/STI (%) - Rural':'Aware_W_HIV_Rural',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of RTI/STI (%) - Urban':'Aware_W_HIV_Urban',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HAF/ORS/ORT/ZINC (%) - Total':'Aware_W_HAF/ORS/ORT/ZINC_Total',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HAF/ORS/ORT/ZINC (%) - Rural':'Aware_W_HAF/ORS/ORT/ZINC_Rural',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of HAF/ORS/ORT/ZINC (%) - Urban':'Aware_W_HAF/ORS/ORT/ZINC_Urban',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of danger signs of ARI/Pneumonia (%) - Total':'Aware_W_ARI_Total',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of danger signs of ARI/Pneumonia (%) - Rural':'Aware_W_ARI_Rural',
                                    'AWARENESS ON HIV/AIDS, RTI/STI, HAF/ORS/ORT/ZINC AND ARI/PNEUMONIA - Women who are aware of danger signs of ARI/Pneumonia (%) - Urban':'Aware_W_ARI_Urban',
                                    'MORTALITY - Crude Death Rate (CDR) - Total - Person':'Mortality_CDR_Total',
                                    'MORTALITY - Crude Death Rate (CDR) - Total - Male':'Mortality_CDR_M_Total',
                                    'MORTALITY - Crude Death Rate (CDR) - Total - Female':'Mortality_CDR_F_Total',
                                    'MORTALITY - Crude Death Rate (CDR) - Rural - Person':'Mortality_CDR_Rural',
                                    'MORTALITY - Crude Death Rate (CDR) - Rural - Male':'Mortality_CDR_M_Rural',
                                    'MORTALITY - Crude Death Rate (CDR) - Rural - Female':'Mortality_CDR_F_Rural',
                                    'MORTALITY - Crude Death Rate (CDR) - Urban - Person':'Mortality_CDR_Urban',
                                    'MORTALITY - Crude Death Rate (CDR) - Urban - Male':'Mortality_CDR_M_Urban',
                                    'MORTALITY - Crude Death Rate (CDR) - Urban - Female':'Mortality_CDR_F_Urban',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Total - Person':'Mortality_IMR_Total',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Total - Male':'Mortality_IMR_M_Total',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Total - Female':'Mortality_IMR_F_Total',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Rural - Person':'Mortality_IMR_Rural',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Rural - Male':'Mortality_IMR_M_Rural',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Rural - Female':'Mortality_IMR_F_Rural',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Urban - Person':'Mortality_IMR_Urban',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Urban - Male':'Mortality_IMR_M_Urban',
                                    'MORTALITY - Infant Mortality Rate (IMR) - Urban - Female':'Mortality_IMR_F_Urban',
                                    'MORTALITY - Neo-natal Mortality Rate - Total':'Mortality_NMR_Total',
                                    'MORTALITY - Neo-natal Mortality Rate - Rural':'Mortality_NMR_Rural',
                                    'MORTALITY - Neo-natal Mortality Rate - Urban':'Mortality_NMR_Urban',
                                    'MORTALITY - Post Neo-natal Mortality Rate - Total':'Mortality_PNMR_Total',
                                    'MORTALITY - Post Neo-natal Mortality Rate - Rural':'Mortality_PNMR_Rural',
                                    'MORTALITY - Post Neo-natal Mortality Rate - Urban':'Mortality_PNMR_Urban',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Total - Person':'Mortality_U5MR_Total',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Total - Male':'Mortality_U5MR_M_Total',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Total - Female':'Mortality_U5MR_F_Total',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Rural - Person':'Mortality_U5MR_Rural',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Rural - Male':'Mortality_U5MR_M_Rural',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Rural - Female':'Mortality_U5MR_F_Rural',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Urban - Person':'Mortality_U5MR_Urban',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Urban - Male':'Mortality_U5MR_M_Urban',
                                    'MORTALITY - Under Five Mortality Rate (U5MR) - Urban - Female':'Mortality_U5MR_F_Urban'})



AHS_data = AHS_data.rename(columns={'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Total - Lower Limit':'CI_CBR_Total_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Total - Upper Limit':'CI_CBR_Total_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Rural - Lower Limit':'CI_CBR_Rural_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Rural - Upper Limit':'CI_CBR_Rural_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Urban - Lower Limit':'CI_CBR_Urban_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Birth Rate - Urban - Upper Limit':'CI_CBR_Urban_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Total - Lower Limit':'CI_CDR_Total_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Total - Upper Limit':'CI_CDR_Total_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Rural - Lower Limit':'CI_CDR_Rural_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Rural - Upper Limit':'CI_CDR_Rural_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Urban - Lower Limit':'CI_CDR_Urban_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Crude Death Rate - Urban - Upper Limit':'CI_CDR_Urban_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Total - Lower Limit':'CI_IMR_Total_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Total - Upper Limit':'CI_IMR_Total_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Rural - Lower Limit':'CI_IMR_Rural_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Rural - Upper Limit':'CI_IMR_Rural_Upper',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Urban - Lower Limit':'CI_IMR_Urban_Lower',
                                    'CONFIDENCE INTERVAL (95%) FOR SOME IMPORTANT INDICATORS - Infant Mortality Rate - Urban - Upper Limit':'CI_IMR_Urban_Upper'})

#------------------------------------------------------------------------------------
Col_names = AHS_data.columns

# impute missing values with 0
#check for Nulls
nullsdata = AHS_data.isna().sum()
print(nullsdata)
AHS_data= AHS_data.replace(np.nan,0)

Urban_Data = pd.DataFrame()
Urban_Data['State'] = AHS_data['State']
Urban_Data['State / District Name'] = AHS_data['State / District Name']
Urban_Col = AHS_data.filter(regex = 'Urban').columns
len(Urban_Col)
Urban_Data = pd.concat([Urban_Data,AHS_data[Urban_Col]],axis=1)
#Urban_Data = Urban_Data.join(AHS_data[Urban_Col])


Total_Data = pd.DataFrame()
Total_Data['State'] = AHS_data['State']
Total_Data['State / District Name'] = AHS_data['State / District Name']
Total_Col = AHS_data.filter(regex = 'Total|total|TOTAL').columns
len(Total_Col)
Total_Data = pd.concat([Total_Data,AHS_data[Total_Col]],axis=1)
#Total_Data = Total_Data.join(AHS_data[Total_Col])


Rural_Data = pd.DataFrame()
Rural_Data['State'] = AHS_data['State']
Rural_Data['State / District Name'] = AHS_data['State / District Name']
Total_Col = AHS_data.filter(regex = 'Rural').columns
len(Total_Col)
AHS_data[Total_Col].columns
Rural_Data = pd.concat([Rural_Data,AHS_data[Total_Col]],axis=1)


def getDuplicateColumns(df): 
  
    # Create an empty set 
    duplicateColumnNames = set() 
      
    # Iterate through all the columns  
    # of dataframe 
    for x in range(df.shape[1]): 
          
        # Take column at xth index. 
        col = df.iloc[:, x] 
          
        # Iterate through all the columns in 
        # DataFrame from (x + 1)th index to 
        # last index 
        for y in range(x + 1, df.shape[1]): 
              
            # Take column at yth index. 
            otherCol = df.iloc[:, y] 
              
            # Check if two columns at x & y 
            # index are equal or not, 
            # if equal then adding  
            # to the set 
            if col.equals(otherCol): 
                duplicateColumnNames.add(df.columns.values[y]) 
                  
    # Return list of unique column names  
    # whose contents are duplicates. 
    return list(duplicateColumnNames) 
  

# Dropping duplicate columns 
Rural_Data = Rural_Data.drop(columns = getDuplicateColumns(Rural_Data)) 

Urban_Data = Urban_Data.drop(columns = getDuplicateColumns(Urban_Data))

Total_Data = Total_Data.drop(columns = getDuplicateColumns(Total_Data))


# Saved to dependent variables in y1 and y2
y1 = Total_Data['ARI_Total']


# deleting dependent variable to get only indepent X features
del_col=['ARI_Total','State','State / District Name']

X = Total_Data.drop(del_col,axis=1)
X.dtypes

# VIF for muticoliinearity check as there r around 200+ features to deal
 
from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] 
  
print(vif_data)


#multicollinearity check
cols = X.columns
totalcols = len(cols)
cor = X.corr()

#plot the heatmap
# difficult to plot the heatmap as there are many features
sns.heatmap(cor,
            xticklabels=cols[0:totalcols-1],
            yticklabels=cols[0:totalcols-1],
            annot=True,
            vmin=0,vmax=1,
            linewidth=0.3,square=True
            )

# finding correlated features 
# above > 0.5
def cor_features(cor):
    correlated_features = set()
    for i in range(len(cor.columns)):
        for j in range(i):
            if abs(cor.iloc[i, j]) > 0.5:
                colname = cor.columns[i]
                correlated_features.add(colname)

    return(correlated_features)

# Drop all correlated columns

X = X.drop(cor_features(cor),axis=1)

# Visualisation/plotting of data using distplot
#distplot for numeric feature

def visualisation_no(f1,f2):
    r=3;c=3;pos=1
    cols = X.columns 
    len(cols)
    fig=plt.figure()
    for e in cols[f1:f2]:
        fig.add_subplot(r,c,pos)
        sns.distplot(X[e],axlabel=e)
        pos+=1
    return

visualisation_no(0,9)
visualisation_no(9,18)
visualisation_no(18,27)
visualisation_no(27,36)
visualisation_no(36,len(X.columns))


# build linear regression model

import statsmodels.api as sm
trainX=sm.add_constant(X)

m1=sm.OLS(y1,trainX).fit()
m1.summary()
'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              ARI_Total   R-squared:                       0.998
Model:                            OLS   Adj. R-squared:                  0.998
Method:                 Least Squares   F-statistic:                     3773.
Date:                Fri, 18 Dec 2020   Prob (F-statistic):               0.00
Time:                        19:45:37   Log-Likelihood:                -1939.6
No. Observations:                 293   AIC:                             3959.
Df Residuals:                     253   BIC:                             4106.
Df Model:                          39                                         
Covariance Type:            nonrobust                                         
==================================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
const                            545.2206    774.855      0.704      0.482    -980.767    2071.208
SU_Total                           0.0061      0.028      0.215      0.830      -0.050       0.062
Housesize_SC_Total              -115.4080     39.091     -2.952      0.003    -192.394     -38.422
Sex_BirthRatio_Total               0.2767      0.283      0.979      0.328      -0.280       0.833
Sex_RatioAll_Total                 0.2997      0.268      1.119      0.264      -0.228       0.827
SC(6-17)_Total                    -4.8905      2.905     -1.683      0.094     -10.612       0.831
Disable_Total                     -0.0437      0.028     -1.541      0.125      -0.100       0.012
Inj_Treated_Total                 -0.0110      0.079     -0.140      0.889      -0.166       0.144
MaInj_Treated_Total                0.1055      0.085      1.235      0.218      -0.063       0.274
MiInj_Treated_Total                0.0081      0.016      0.506      0.613      -0.024       0.040
Diarrhoea_Total                   -0.0236      0.020     -1.156      0.249      -0.064       0.017
ARI_M_Total                        1.0335      0.004    253.620      0.000       1.025       1.042
Fever_Total                       -0.0015      0.005     -0.309      0.758      -0.011       0.008
Any_Treatment_Total               -0.3223      1.780     -0.181      0.856      -3.827       3.182
CI_AnySym_Total                   -0.0046      0.004     -1.219      0.224      -0.012       0.003
CI_SMed_Total                      1.9938      1.909      1.044      0.297      -1.766       5.754
CI_TB_Total                       -0.0978      0.132     -0.738      0.461      -0.359       0.163
CI_Asthma_Total                    0.0346      0.037      0.946      0.345      -0.037       0.107
CI_RT_Total                       -2.5177      1.543     -1.632      0.104      -5.556       0.520
W(15-19)_Pre_Total                 2.3535      1.671      1.408      0.160      -0.937       5.644
LB_Aft36m_Total                   -3.1971      2.164     -1.478      0.141      -7.459       1.064
Abort_ANCRec_Total                 0.9342      1.291      0.724      0.470      -1.608       3.476
Abort_USDOne_Total                -0.8386      1.552     -0.540      0.590      -3.896       2.219
Abort_SHP_Total                    0.2712      0.975      0.278      0.781      -1.649       2.191
FP_MSter_Total                    -8.8847     11.908     -0.746      0.456     -32.336      14.567
FP_IUD_Total                      -1.7849     14.698     -0.121      0.903     -30.731      27.162
FP_Condom_Total                    4.6002      2.911      1.580      0.115      -1.132      10.333
FP_ECPills_Total                  -2.9713     40.529     -0.073      0.942     -82.788      76.845
DC_PrivInstDel_Total               3.7386      2.029      1.843      0.067      -0.257       7.735
IMM_recIFA3mths_Total             -2.5181      1.540     -1.636      0.103      -5.550       0.514
IMM_BirthWeight<2.5_Total          1.6708      2.095      0.797      0.426      -2.456       5.798
CD_Diarrhoea_Total                -0.2513      2.076     -0.121      0.904      -4.339       3.837
CD_DiaRecORS_Total                 1.2832      1.452      0.883      0.378      -1.577       4.144
CD_ARITreated_Total               -0.2645      3.011     -0.088      0.930      -6.193       5.664
CD_FeverTreated_Total             -0.9238      2.949     -0.313      0.754      -6.732       4.885
BnS_bffor6mths_Total               2.6191      1.103      2.375      0.018       0.448       4.791
BnS_bfnSSfoodAft6mths_Total       -0.8664      2.512     -0.345      0.730      -5.814       4.081
Aware_W_STI_Total                 -1.6039      0.891     -1.801      0.073      -3.358       0.150
Aware_W_HAF/ORS/ORT/ZINC_Total     0.8571      4.333      0.198      0.843      -7.676       9.391
Mortality_CDR_Total               -9.6833     11.804     -0.820      0.413     -32.931      13.564
==============================================================================
Omnibus:                       25.155   Durbin-Watson:                   2.206
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               63.489
Skew:                           0.372   Prob(JB):                     1.63e-14
Kurtosis:                       5.156   Cond. No.                     9.14e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.14e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

'''
y2 = Total_Data['Fever_Total']

del_col=['Fever_Total','State','State / District Name']

X = Total_Data.drop(del_col,axis=1)
X.dtypes

cols = X.columns
totalcols = len(cols)
cor = X.corr()

# Drop all correlated columns

X = X.drop(cor_features(cor),axis=1)

#Plots/Visualisation of data
visualisation_no(0,9)
visualisation_no(9,18)
visualisation_no(18,27)
visualisation_no(27,36)
visualisation_no(36,len(X.columns))


# building OLS model
import statsmodels.api as sm
trainX=sm.add_constant(X)
m_y2 = sm.OLS(y2,trainX).fit()
m_y2.summary()

'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            Fever_Total   R-squared:                       0.995
Model:                            OLS   Adj. R-squared:                  0.994
Method:                 Least Squares   F-statistic:                     1268.
Date:                Sun, 20 Dec 2020   Prob (F-statistic):          7.70e-268
Time:                        17:02:32   Log-Likelihood:                -2020.5
No. Observations:                 293   AIC:                             4121.
Df Residuals:                     253   BIC:                             4268.
Df Model:                          39                                         
Covariance Type:            nonrobust                                         
==================================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------------
const                           -671.4671   1021.136     -0.658      0.511   -2682.477    1339.542
SU_Total                          -0.0198      0.037     -0.530      0.596      -0.094       0.054
Housesize_SC_Total                24.4512     51.402      0.476      0.635     -76.779     125.682
Sex_BirthRatio_Total               0.0577      0.372      0.155      0.877      -0.676       0.791
Sex_RatioAll_Total                 0.9600      0.353      2.716      0.007       0.264       1.656
SC(6-17)_Total                     2.7548      3.832      0.719      0.473      -4.793      10.302
Disable_Total                      0.0103      0.037      0.277      0.782      -0.063       0.084
Inj_Treated_Total                 -0.2469      0.104     -2.379      0.018      -0.451      -0.043
MaInj_Treated_Total                0.0753      0.113      0.668      0.505      -0.147       0.297
MiInj_Treated_Total                0.0319      0.021      1.507      0.133      -0.010       0.074
Diarrhoea_Total                    0.0625      0.027      2.336      0.020       0.010       0.115
ARI_Total                         -0.0050      0.005     -0.961      0.338      -0.015       0.005
Fever_M_total                      1.0268      0.006    159.039      0.000       1.014       1.040
Any_Treatment_Total               -1.9948      2.346     -0.850      0.396      -6.615       2.626
CI_AnySym_Total                   -0.0081      0.005     -1.622      0.106      -0.018       0.002
CI_SMed_Total                     -6.0815      2.511     -2.422      0.016     -11.028      -1.136
CI_TB_Total                       -0.0738      0.175     -0.423      0.673      -0.418       0.270
CI_Asthma_Total                    0.0705      0.048      1.462      0.145      -0.024       0.166
CI_RT_Total                       -2.6295      2.032     -1.294      0.197      -6.632       1.373
W(15-19)_Pre_Total                 0.0785      2.202      0.036      0.972      -4.259       4.416
LB_Aft36m_Total                    0.2533      2.853      0.089      0.929      -5.365       5.872
Abort_ANCRec_Total                 1.1044      1.700      0.650      0.517      -2.244       4.452
Abort_USDOne_Total                -2.4682      2.045     -1.207      0.229      -6.495       1.559
Abort_SHP_Total                    1.6028      1.285      1.247      0.214      -0.929       4.134
FP_MSter_Total                     0.0587     15.691      0.004      0.997     -30.843      30.960
FP_IUD_Total                       8.5729     19.370      0.443      0.658     -29.573      46.719
FP_Condom_Total                    9.9361      3.838      2.589      0.010       2.377      17.495
FP_ECPills_Total                 -81.6680     53.450     -1.528      0.128    -186.931      23.595
DC_PrivInstDel_Total               0.4248      2.677      0.159      0.874      -4.847       5.696
IMM_recIFA3mths_Total             -2.5054      2.030     -1.234      0.218      -6.503       1.492
IMM_BirthWeight<2.5_Total         -2.8878      2.763     -1.045      0.297      -8.329       2.554
CD_Diarrhoea_Total                -1.9311      2.736     -0.706      0.481      -7.319       3.456
CD_DiaRecORS_Total                 4.3025      1.915      2.247      0.026       0.531       8.074
CD_ARITreated_Total               -3.8963      3.968     -0.982      0.327     -11.710       3.918
CD_FeverTreated_Total              3.1908      3.886      0.821      0.412      -4.462      10.844
BnS_bffor6mths_Total              -2.4525      1.452     -1.689      0.092      -5.312       0.407
BnS_bfnSSfoodAft6mths_Total        0.4666      3.311      0.141      0.888      -6.054       6.987
Aware_W_STI_Total                 -2.0799      1.173     -1.773      0.077      -4.391       0.231
Aware_W_HAF/ORS/ORT/ZINC_Total     3.7091      5.707      0.650      0.516      -7.531      14.949
Mortality_CDR_Total               -3.1606     15.547     -0.203      0.839     -33.778      27.457
==============================================================================
Omnibus:                       33.436   Durbin-Watson:                   1.927
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               80.832
Skew:                           0.542   Prob(JB):                     2.80e-18
Kurtosis:                       5.334   Cond. No.                     9.10e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 9.1e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

'''