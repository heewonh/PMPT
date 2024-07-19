# Import libraries
import pandas as pd
import itertools
import math
from scipy.optimize import minimize
from scipy import integrate
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Changing directory to PMPT folder
os.chdir('/Users/heewon/Dev/Projects/Quant/PMPT')

# Activating program
def main():
    # Input analysis period
    period = input("Enter analysis period ('total' / in months): ").lower()

    # error handling
    while period != 'total':
        try:
            period = int(period)
        except ValueError:
            print("Period input is invalid")
            period = input("Enter analysis period ('total' / in months): ").lower()
            continue
        period = int(period)
        break
    
    # Input return interval
    rtn_interval = int(input("Enter return interval of dataset (1 / 3 / 12): "))
    
    # error handling
    while rtn_interval not in [1, 3, 12]:
        print("Invalid return interval")
        rtn_interval = int(input("Enter return interval of dataset (1 / 3 / 12): "))

    # DB GAPS port
    port = ['KODEX 200',
            'TIGER KOSDAQ150',
            'TIGER S&P500',
            'TIGER SYNTH-EURO STOXX 50(H)',
            'ACE Japan Nikkei225(H)',
            'TIGER CHINA A300',
            'KOSEF 10YKTB',
            'KBSTAR Credit',
            'TIGER SYNTH-HY(H)',
            'KODEX Gold Futures(H)',
            'TIGER WTI Futures',
            'KODEX INVERSE',
            'KODEX USD Futures',
            'KODEX USD Futures Inverse',
            'KOSEF Enhanced Cash']
    
    # Determine risk-free rate
    rtn_df = pd.read_csv(f'data/3yr_{rtn_interval}mo.csv')
    rtn_df = rtn_df.reset_index().drop(columns=['index', 'D A T E'])
    
    if period != 'total':
        del_index = period * 30  # rows 1 ~ del_index deleted from rtn_df (assuming 30 days in 1 mo)
        rtn_df = rtn_df.drop(range(0, len(rtn_df.index) - del_index))
    
    avg_rtn = rtn_df.mean()
    rf = avg_rtn['KOSEF Enhanced Cash']
    
    # Input MAR
    mar = int(input("Enter MAR (annual rtn _%): "))
    mar = mar / 100 + 1
    
    # Initializing class
    pmpt = PMPT(period=period, rtn_interval=rtn_interval, port=port, rf=rf, mar=mar) 
    
    # Setup initial weights (equal weights)
    weights = []
    init_weight = 0.99 / len(port)
    for _ in range(len(port)):
        weights.append(init_weight)
    
    # Running functions
    pmpt.data_sampling()
    pmpt.hist_plot()
    pmpt.integrate_lognormal(weights)  # should result = 1
    pmpt.lognormal_plot(weights)
    
    # Constraints list
    constraints = [
        {'type': 'ineq', 'fun': pmpt.kospi_ceil},
        {'type': 'ineq', 'fun': pmpt.kospi_floor},
        {'type': 'ineq', 'fun': pmpt.kosdaq_ceil},
        {'type': 'ineq', 'fun': pmpt.kosdaq_floor},
        {'type': 'ineq', 'fun': pmpt.kor_stock_ceil},
        {'type': 'ineq', 'fun': pmpt.kor_stock_floor},
        {'type': 'ineq', 'fun': pmpt.sp500_ceil},
        {'type': 'ineq', 'fun': pmpt.sp500_floor},
        {'type': 'ineq', 'fun': pmpt.stoxx50_ceil},
        {'type': 'ineq', 'fun': pmpt.stoxx50_floor},
        {'type': 'ineq', 'fun': pmpt.nikkei225_ceil},
        {'type': 'ineq', 'fun': pmpt.nikkei225_floor},
        {'type': 'ineq', 'fun': pmpt.csi300_ceil},
        {'type': 'ineq', 'fun': pmpt.csi300_floor},
        {'type': 'ineq', 'fun': pmpt.for_stock_ceil},
        {'type': 'ineq', 'fun': pmpt.for_stock_floor},
        {'type': 'ineq', 'fun': pmpt.bond_10yr_ceil},
        {'type': 'ineq', 'fun': pmpt.bond_10yr_floor},
        {'type': 'ineq', 'fun': pmpt.corp_bond_ceil},
        {'type': 'ineq', 'fun': pmpt.corp_bond_floor},
        {'type': 'ineq', 'fun': pmpt.for_bond_ceil},
        {'type': 'ineq', 'fun': pmpt.for_bond_floor},
        {'type': 'ineq', 'fun': pmpt.bond_ceil},
        {'type': 'ineq', 'fun': pmpt.bond_floor},
        {'type': 'ineq', 'fun': pmpt.gold_ceil},
        {'type': 'ineq', 'fun': pmpt.gold_floor},
        {'type': 'ineq', 'fun': pmpt.wti_ceil},
        {'type': 'ineq', 'fun': pmpt.wti_floor},
        {'type': 'ineq', 'fun': pmpt.comm_ceil},
        {'type': 'ineq', 'fun': pmpt.comm_floor},
        {'type': 'ineq', 'fun': pmpt.kospi_short_ceil},
        {'type': 'ineq', 'fun': pmpt.kospi_short_ceil2},
        {'type': 'ineq', 'fun': pmpt.kospi_short_floor},
        {'type': 'ineq', 'fun': pmpt.us_long_ceil},
        {'type': 'ineq', 'fun': pmpt.us_long_floor},
        {'type': 'ineq', 'fun': pmpt.us_short_ceil},
        {'type': 'ineq', 'fun': pmpt.us_short_floor},
        {'type': 'ineq', 'fun': pmpt.dollar_ceil},
        {'type': 'ineq', 'fun': pmpt.mmf_ceil},
        {'type': 'ineq', 'fun': pmpt.mmf_floor},
        {'type': 'eq', 'fun': pmpt.sum_weights}
    ]
    
    # Optimization results
    result = minimize(pmpt.objective, weights, constraints=constraints)
    print(result)
    print(result.x)
    
    # Downward deviation
    print(f'Downside deviation: {pmpt.downside_dev(weights)}')


class PMPT:
    def __init__(self, period, rtn_interval, port, rf, mar):
        self.period = period  # string 2mo, 3mo, 12mo, total etc.
        self.rtn_interval = rtn_interval # int 3 monthly rtn etc.
        self.port = port  # string list of tickers
        self.rf = rf  # risk-free rate
        self.mar = mar # MAR (minimum acceptable return)
    
    # Asset data sampling: random sampling -> get mean, stdv, extreme vals for each asset data
    def data_sampling(self):
        # read rtn data from Quantiwise csv
        rtn_df = pd.read_csv(f'data/3yr_{self.rtn_interval}mo.csv')
        
        # getting rid of unnecessary cols
        rtn_df = rtn_df.reset_index().drop(columns=['index', 'D A T E'])
        
        # trimming rtn_df to match period (that isn't total period)
        if self.period != 'total':
            del_index = self.period * 30  # rows 1 ~ del_index deleted from rtn_df (assuming 30 days in 1 mo)
            rtn_df = rtn_df.drop(range(0, len(rtn_df.index) - del_index))
        
        # random sampling rtn data of each asset
        sampled_df = pd.DataFrame(columns=self.port)  # df of annualized returns calculated from randomly sampled returns
        sample_size = int(12 / self.rtn_interval)  # sample size based on rtn_interval ex: 3 monthly rtn -> size = 4

        for _ in range(2500):  # book did random sampling 2,500 times
            temp_df = pd.DataFrame()
            
            for column_name, column_data in rtn_df.items():
                sampled_data = column_data.sample(n=sample_size, replace=True) + 1
                temp_df[column_name] = sampled_data.values
                    
            # compound returns in each col
            temp_result = temp_df.prod()
            
            # add data in temp_result to sampled_df
            sampled_df.loc[len(sampled_df)] = temp_result
        
        # Output results
        cwd = os.getcwd()
        sampled_df.to_csv(f'{cwd}/data/sampled_rtn.csv', index=False)
        
        # calculate & output average return (expected return)
        avg_rtn = sampled_df.mean()
        avg_rtn.to_csv(f'{cwd}/data/avg_rtn.csv')

        # calculate & output std of weekly return
        std_df = sampled_df.std()
        std_df.to_csv(f'{cwd}/data/std.csv')

        # calculate & output correlation matrix of weekly return
        corr_df = rtn_df.corr()  # notice that this uses rtn_df
        corr_df.to_csv(f'{cwd}/data/corr.csv')
        
    # plotting the random annualized rtn for each asset
    def hist_plot(self):
        sampled_df = pd.read_csv('data/sampled_rtn.csv')

        for col_name in sampled_df.columns:    
            plt.figure()  # Create a new figure for each histogram
            plt.hist(sampled_df[col_name], bins=50)  # Adjust the number of bins as needed
            plt.title(f'Histogram of {col_name}')
            plt.xlabel(col_name)
            plt.ylabel('Frequency')
        plt.show()  # Display all histograms
            
    # Calculating E(r) / mean of port
    def port_exp_rtn(self, weights):
        df = pd.read_csv('data/avg_rtn.csv')
        result = 0
        
        for index, row in df.iterrows():
            result += weights[index] * row['0']
            
        return result
    
    # Calculating std of port
    def port_std(self, weights):
        std_df = pd.read_csv('data/std.csv')
        corr_df = pd.read_csv('data/corr.csv')
        var_first = 0
        var_second = 0
        
        # first half of variance
        for index, row in std_df.iterrows():
            var_first += (weights[index]**2 * row['0']**2)
        
        # second half of variance
        variables = []
        counter = 0
        
        for item in self.port:
            variables.append(counter)
            counter += 1
            
        combinations = list(itertools.combinations(variables, 2))
        
        for combination in combinations:
            a = combination[0]
            b = combination[1]
            var_second += weights[a]*weights[b] * corr_df.iloc[a,b+1] * std_df.iloc[a,1]*std_df.iloc[b,1]
        
        var = var_first + 2 * var_second
        std = math.sqrt(var)
        return std

    # Calculating extreme values for port lognormal
    def port_ext_val(self, weights):
        df = pd.read_csv('data/sampled_rtn.csv')
        min_series = df.min()
        max_series = df.max()
        min = 0
        max = 0
        result = 0
        result_isMin = True  
        
        # min calculation
        for index, min_val in enumerate(min_series):
            min += weights[index] * min_val
            
        # max calculation
        for index, max_val in enumerate(max_series):
            max += weights[index] * max_val
            
        # extreme value determination (extreme value is min/max - whichever is closer to the mean - moved away from the mean by 4 std)
        min_diff = abs(self.port_exp_rtn(weights) - min)
        max_diff = abs(self.port_exp_rtn(weights) - max)
        
        if min_diff <= max_diff:  # when min_diff == max_diff, follow min since expected return generally follows right skew
            result = min - 4 * self.port_std(weights)
            result_isMin = True  # extreme val is found with min
        else:
            result = max + 4 * self.port_std(weights)
            result_isMin = False  # extreme val is found with max
            
        return result, result_isMin
    
    # Calculating formula for port lognormal curve
    def port_lognormal(self, weights):
        mean = self.port_exp_rtn(weights)
        std = self.port_std(weights)
        tau, ext_isMin = self.port_ext_val(weights)  # extreme val, ext_val_isMin
        
        # auxiliary parameters
        dif = abs(mean - tau)
        sig = math.log((std / dif) ** 2 + 1)
        mu = math.log(dif) - sig ** 2
        alpha = 1 / (sig * math.sqrt(2 * math.pi))
        beta = -1 / (2 * (sig ** 2))
        
        # formula for lognormal
        if ext_isMin:
            func = lambda x: alpha / (x - tau) * math.exp(beta * (math.log(x - tau) - mu) ** 2)
        else:
            func = lambda x: alpha / (tau - x) * math.exp(beta * (math.log(tau - x) - mu) ** 2)
        
        return func, tau, ext_isMin
    
    # Graphing the lognormal curve 
    def lognormal_plot(self, weights):
        func, tau, ext_isMin = self.port_lognormal(weights)
        func_vec = np.vectorize(func)
        x = np.linspace(1, 1.125, num=100)  # hard-coded range
        y = func_vec(x)
        
        plt.plot(x, y, label='lognormal func')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plot of a Lognormal')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Lognormal integration check (should = 1)
    def integrate_lognormal(self, weights):
        func, tau, ext_isMin = self.port_lognormal(weights)
        
        if ext_isMin:
            lower_bound = tau
            upper_bound = np.inf
        else:
            lower_bound = -np.inf
            upper_bound = tau
        
        result, error = integrate.quad(func, lower_bound, upper_bound)
        
        if result < 0.5:
            print(f'Error: Integral of lognormal ({result}) is far less than 1')
            sys.exit()
        else:
            print(f'Integration success! Result: {result}, Error: {error}')
        
    # Downside deviation
    def downside_dev(self, weights):
        func, tau, ext_isMin = self.port_lognormal(weights)
        
        upper_bound = self.mar
        if ext_isMin:        
            lower_bound = tau
        else:
            lower_bound = -np.inf
        
        integrate_func = lambda x: (self.mar - x) ** 2 * func(x)
        integral, error = integrate.quad(integrate_func, lower_bound, upper_bound)
        dd = math.sqrt(integral)
        return dd
    
    # Optimization
    # Objective : maximize Sortino ratio
    # sign = -1.0 used since scipy.optimize only has minimize func (hence need to find min of the negative of our objective func)
    def objective(self, x, sign=-1.0):
        return sign * ((self.port_exp_rtn(weights=x) - self.rf) / self.downside_dev(weights=x))

    # Constraints formating (=> 0) :
    # Let x[0] be weight of KOSPI and let its constraint be that KOSPI <= 10%
    # - KOSPI + 0.1 => 0
    # => -x[0] + 0.1
    
    # Constraint 1 : KOSPI ceiling (x0 <= 0.4)
    def kospi_ceil(self, x):
        return -x[0] + 0.4
    # Constraint 2 : KOSPI floor (x0 => 0)
    def kospi_floor(self, x):
        return x[0]
    # Constraint 3 : KOSDAQ ceiling (x1 <= 0.2)
    def kosdaq_ceil(self, x):
        return -x[1] + 0.2
    # Constraint 4 : KOSDAQ floor (x1 => 0)
    def kosdaq_floor(self, x):
        return x[1]
    # Constraint 5 : 국내주식 ceiling (x0 + x1 <= 0.4)
    def kor_stock_ceil(self, x):
        return -x[0] - x[1] + 0.4
    # Constraint 6 : 국내주식 floor (x0 + x1 >= 0.1)
    def kor_stock_floor(self, x):
        return x[0] + x[1] - 0.1
    # Constraint 7 : S&P 500 ceiling (x2 <= 0.2)
    def sp500_ceil(self, x):
        return -x[2] + 0.2
    # Constraint 8 : S&P 500 floor (x2 => 0)
    def sp500_floor(self, x):
        return x[2]
    # Constraint 9 : STOXX 50 ceiling (x3 <= 0.2)
    def stoxx50_ceil(self, x):
        return -x[3] + 0.2
    # Constraint 10 : STOXX 50 floor (x3 => 0)
    def stoxx50_floor(self, x):
        return x[3]
    # Constraint 11 : Nikkei 225 ceiling (x4 <= 0.2)
    def nikkei225_ceil(self, x):
        return -x[4] + 0.2
    # Constraint 12 : Nikkei 225 floor (x4 => 0)
    def nikkei225_floor(self, x):
        return x[4]
    # Constraint 13 : CSI 300 ceiling (x5 <= 0.2)
    def csi300_ceil(self, x):
        return -x[5] + 0.2
    # Constraint 14 : CSI 300 floor (x5 => 0)
    def csi300_floor(self, x):
        return x[5]
    # Constrait 15 : 해외주식 ceiling (x2 + x3 + x4 + x5 <= 0.4)
    def for_stock_ceil(self, x):
        return -x[2] - x[3] -x[4] - x[5] + 0.4
    # Constraint 16 : 해외주식 floor (x2 + x3 + x4 + x5 => 0.1)
    def for_stock_floor(self, x):
        return x[2] + x[3] + x[4] + x[5] - 0.1
    # Constraint 17 : 국채 10년 ceiling (x6 <= 0.5)
    def bond_10yr_ceil(self, x):
        return -x[6] + 0.5
    # Constraint 18 : 국채 10년 floor (x6 >= 0)
    def bond_10yr_floor(self, x):
        return x[6]
    # Constraint 19 : 우량회사채 ceiling (x7 <= 0.4)
    def corp_bond_ceil(self, x):
        return -x[7] + 0.4
    # Constraint 20 : 우랭회사채 floor (x7 >= 0)
    def corp_bond_floor(self, x):
        return x[7]
    # Constraint 21 : 해외채권 ceiling (x8 <= 0.4)
    def for_bond_ceil(self, x):
        return -x[8] + 0.4
    # Constraint 22 : 해외채권 floor (x8 >= 0.05)
    def for_bond_floor(self, x):
        return x[8] - 0.05
    # Constraint 23 : bond ceiling (x6 + x7 + x8 <= 0.6)
    def bond_ceil(self, x):
        return -x[6] - x[7] -x[8] + 0.6
    # Constraint 24 : bond floor (x6 + x7 + x8 => 0.2)
    def bond_floor(self, x):
        return x[6] + x[7] + x[8] - 0.2
    # Constraint 25 : gold ceiling (x9 <= 0.15)
    def gold_ceil(self, x):
        return -x[9] + 0.15
    # Constraint 26 : gold floor (x9 => 0)
    def gold_floor(self, x):
        return x[9]
    # Constraint 27 : WTI ceiling (x10 <= 0.15)
    def wti_ceil(self, x):
        return -x[10] + 0.15
    # Constraint 28 : WTI floor (x10 => 0)
    def wti_floor(self, x):
        return x[10]
    # Constraint 29 : commodities ceiling (x9 + x10 <= 0.2)
    def comm_ceil(self, x):
        return -x[9] - x[10] + 0.2
    # Constraint 30 : commodities floor (x9 + x10 => 0.05)
    def comm_floor(self, x):
        return x[9] + x[10] - 0.05
    # Constraint 31 : KOSPI short ceiling (x11 <= 0.2)
    def kospi_short_ceil(self, x):
        return -x[11] + 0.2
    # Constraint 32 : KOSPI short ceiling 2 (x11 <= x0 + x1 --> x0 + x1 -x11 => 0)
    def kospi_short_ceil2(self, x):
        return x[0] + x[1] - x[11]
    # Constraint 33 : KOSPI short floor (x11 => 0)
    def kospi_short_floor(self, x):
        return x[11]
    # Constraint 34 : US long ceiling (x12 <= 0.2)
    def us_long_ceil(self, x):
        return -x[12] + 0.2
    # Constraint 35 : US long floor (x12 => 0)
    def us_long_floor(self, x):
        return x[12]
    # Constraint 36 : US short ceiling (x13 <= 0.2)
    def us_short_ceil(self, x):
        return -x[13] + 0.2
    # Constraint 37 : US short floor (x13 => 0)
    def us_short_floor(self, x):
        return x[13]
    # Constraint 38 : US dollar ceiling (x12 + x13 <= 0.2)
    def dollar_ceil(self, x):
        return -x[12] - x[13] + 0.2
    # Constraint 39 : MMF ceiling (x14 <= 0.49)
    def mmf_ceil(self, x):
        return -x[14] + 0.49
    # Constraint 40 : MMF floor (x14 => 0)
    def mmf_floor(self, x):
        return x[14]
    # Constraint 41 : sum of weights (x0 + x1 + ... + x14 = 0.99)
    def sum_weights(self, x):
        return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] - 0.99


if __name__ == "__main__":
    main()
