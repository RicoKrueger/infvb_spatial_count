import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, KNN
import pickle

###
#Load data
###

df = pd.read_csv('bym_nyc_study.csv')

###
#Load and process supplementary data
###

df_supp_list = [pd.read_csv('supp_data/{}.csv'.format(d), header=2) 
           for d in ['total_population', 'race', 'poverty', 'workers']]

def to_int(x):
    n = ''
    for c in x:
        if c != ',':
            n += c
    return int(n)

def clean(d):
    d.drop(columns=d.columns[-1], inplace=True)
    d = d[d['Output']=='Estimate'].copy()
    d.drop(columns='Output', inplace=True)
    d.iloc[:,-1] = d.iloc[:,-1].apply(to_int)
    return d

df_supp_list = list(map(clean, df_supp_list))

df_supp_list[0].set_index('RESIDENCE', inplace=True)
df_supp_list[1] = df_supp_list[1].pivot(
    index='RESIDENCE', columns='Race of Person 5', values='Population'
    )
df_supp_list[2] = df_supp_list[2].pivot(
    index='RESIDENCE', columns='Poverty Status 4', 
    values='Households with Poverty Status'
    )
df_supp_list[3].set_index('WORKPLACE',inplace=True)

df_supp = df_supp_list[0].copy()
for d in df_supp_list[1:]:
    df_supp = df_supp.join(d, how='inner')
    
df_supp.reset_index(inplace=True)
df_supp.rename(columns={'index': 'census_tract'}, inplace=True)

def extract_ctlabel_county(x):
    ctlabel = ''
    for c in x[(len('Census Tract ')):]:
        if c == ',':
            break
        ctlabel += c
    
    county = ''
    i0 = len('Census Tract ') + len(ctlabel) + 2
    for i, c in enumerate(x[i0:], start=i0):
        if x[i:(i+len(' County'))] == ' County':
            break
        county += c
        
    return pd.Series(data={'ctlabel': ctlabel, 'county': county})

df_supp[['ctlabel', 'county']] = df_supp['census_tract']\
    .apply(extract_ctlabel_county)
df_supp['boro_name'] = df_supp['county'].replace({'New York': 'Manhattan',
                                                  'Kings': 'Brooklyn',
                                                  'Richmond': 'Staten Island'})
df_supp = df_supp[df_supp['boro_name'].isin(
    ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']
    )]

###
#Load geometry
###

shp_path = '2010 Census Tracts/' \
    + 'geo_export_aadf299c-553e-4b51-ba18-7adf52b70243.shp'
geo = gpd.read_file(shp_path)

###
#Rename census tract labels
###

df['census_tract'] = df['census_tract'].astype(str).str[3:]

#61 -> Manhattan -> 1
def rename(x):
    if len(x) == 8 and x[:2] == '61':
        return '1' + x[2:]
    else:
        return x
df['census_tract'] = df['census_tract'].apply(rename)

#05 -> Bronx -> 2
def rename(x):
    if len(x) == 8 and x[:2] == '05':
        return '2' + x[2:]
    else:
        return x
df['census_tract'] = df['census_tract'].apply(rename)

#47 -> Brooklyn -> 3
def rename(x):
    if len(x) == 8 and x[:2] == '47':
        return '3' + x[2:]
    else:
        return x
df['census_tract'] = df['census_tract'].apply(rename)

#81 -> Queens -> 4
def rename(x):
    if len(x) == 8 and x[:2] == '81':
        return '4' + x[2:]
    else:
        return x
df['census_tract'] = df['census_tract'].apply(rename)

#85 -> Staten Island -> 5
def rename(x):
    if len(x) == 8 and x[:2] == '85':
        return '5' + x[2:]
    else:
        return x
df['census_tract'] = df['census_tract'].apply(rename)

###
#Join data and geometry, select subsample
###

geo = geo.rename(columns={'boro_ct201': 'census_tract'})
geo['census_tract'] = geo['census_tract'].astype(str)
geo = geo.merge(df, on='census_tract', how='right')

geo['ct_aux'] = geo['boro_name'] + '_' + geo['ctlabel']
df_supp['ct_aux'] = df_supp['boro_name'] + '_' + df_supp['ctlabel']
df_supp.drop(columns=['census_tract', 'boro_name', 'ctlabel'], inplace=True)
geo = geo.merge(df_supp, on='ct_aux', how='left')
geo.drop(columns=['ct_aux', 'county'], inplace=True)

geo.sort_values(by=['census_tract'], inplace=True)

geo = geo[geo['boro_name'].isin(['Manhattan', 'Bronx'])].copy()

###
#Process attributes
###

geo['black'] = geo['Black or African American alone'] / geo['Population']
geo['poor'] = geo['Below 100 percent of the poverty level'] \
    / geo['Total, poverty status']

geo = geo.to_crs(epsg=6933)
geo['area_km2'] = geo.area / 1000**2
geo['workers_per_km2'] = geo['Workers 16 and Over'] / geo['area_km2']

geo.drop(
    columns=
    ['Population',
     'All Other, i.e., 2 or more races, Native Hawaiian or Pacific Islander, American Indian or Alaska Native, Other race',
     'All races', 'Asian alone', 'Black or African American alone',
     'White alone', '100 to 149 percent of the poverty level',
     'At or above 150 percent of the poverty level',
     'Below 100 percent of the poverty level', 'Total, poverty status',
     'Workers 16 and Over'
     ], 
    inplace=True)

###
#Create spatial weights matrix
###

"""
#Queens contiguity
q_c = Queen.from_dataframe(geo)
C = q_c.full()[0]

#Manually connect islands
#geo.iloc[q_c.islands[0]]
islands_ct = [geo.iloc[i]['census_tract'] for i in q_c.islands]

#Connect 1030900 to 1029900
if '1030900' in islands_ct:
    own_idx = geo['census_tract'] == '1030900'
    new_idx = geo['census_tract'] == '1029900'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1

#Connect 2051600 to 2046201
if '2051600' in islands_ct:
    own_idx = geo['census_tract'] == '2051600'
    new_idx = geo['census_tract'] == '2046201'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1

#Connect 4091601 to 4092200
if '4091601' in islands_ct:
    own_idx = geo['census_tract'] == '4091601'
    new_idx = geo['census_tract'] == '4092200'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1
    
#Connect 4107201 to 4089200, 4094201, 4094202
if '4107201' in islands_ct:
    own_idx = geo['census_tract'] == '4107201'
    
    new_idx = geo['census_tract'] == '4089200'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1
    
    new_idx = geo['census_tract'] == '4094201'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1
    
    new_idx = geo['census_tract'] == '4094202'
    C[own_idx, new_idx] = 1
    C[new_idx, own_idx] = 1
"""

#K-nearest neighbour
q_k = KNN.from_dataframe(geo, k=8)
C = q_k.full()[0]

#Row-normalised weight matrix
W = C / C.sum(axis=1)

x = C.sum(axis=1)
print(x[x <= 1])

###
#Store
###

data = geo[[*df.columns, 'black', 'poor', 'workers_per_km2']]
print(data.isna().sum())
data.to_csv('crash_data.csv', index=False)

filename = 'weight_matrix'
outfile = open(filename, 'wb')
pickle.dump(W, outfile)
outfile.close()

filename = 'crash_data_geo'
outfile = open(filename, 'wb')
pickle.dump(geo, outfile)
outfile.close()