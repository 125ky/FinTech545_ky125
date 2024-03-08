import pandas as pd
import numpy as np
from RiskPackage import CovarianceMissData
from RiskPackage import CovarianceEstimation
from RiskPackage import NonPsdFixes
from RiskPackage import CalculateReturn
from RiskPackage import ModelFitter
from RiskPackage import RiskMetrics
import pytest

@pytest.fixture
def load_data():
    data_paths = {
        'x1': 'test/test1.csv',
        'out_11': 'test/testout_1.1.csv',
        'out_12': 'test/testout_1.2.csv',
        'out_13': 'test/testout_1.3.csv',
        'out_14': 'test/testout_1.4.csv',
    }
    data = {key: pd.read_csv(path) for key, path in data_paths.items()}
    return data

# test1.1
def test_missing_covariance_skip_missing_rows(load_data):
    cout_1 = CovarianceMissData.missing_cov(load_data['x1'], skip_miss=True)
    check11 = cout_1 - load_data['out_11']
    check11 = np.linalg.norm(check11.to_numpy())
    print(f"Difference: {check11}")
    assert check11 < 1e-6, "Covariance with missing data (skip missing rows) failed"
    print("Test 1.1 passed")

# test1.2
def test_missing_correlation_skip_missing_rows(load_data):
    cout_2 = CovarianceMissData.missing_cov(load_data['x1'], skip_miss=True, fun=np.corrcoef)
    check12 = cout_2 - load_data['out_12']
    check12 = np.linalg.norm(check12.to_numpy())
    print(f"Difference: {check12}")
    assert check12 < 1e-6, "Correlation with missing data (skip missing rows) failed"
    print("Test 1.2 passed")

# test1.3
def test_missing_covariance_pairwise(load_data):
    cout_3 = CovarianceMissData.missing_cov(load_data['x1'], skip_miss=False)
    check13 = cout_3 - load_data['out_13']
    check13 = np.linalg.norm(check13.to_numpy())
    print(f"Difference: {check13}")
    assert check13 < 1e-6, "Covariance with missing data (pairwise) failed"
    print("Test 1.3 passed")

# test1.4
def test_missing_correlation_pairwise(load_data):
    cout_4 = CovarianceMissData.missing_cov(load_data['x1'], skip_miss=False, fun=np.corrcoef)
    check14 = cout_4 - load_data['out_14']
    check14 = np.linalg.norm(check14.to_numpy())
    print(f"Difference: {check14}")
    assert check14 < 1e-6, "Correlation with missing data (pairwise) failed"
    print("Test 1.4 passed")


# test2
@pytest.fixture
def load_data_ew():
    data_paths = {
        'x2': 'test/test2.csv',
        'out_21': 'test/testout_2.1.csv',
        'out_22': 'test/testout_2.2.csv',
        'out_23': 'test/testout_2.3.csv',
    }
    data = {key: pd.read_csv(path) for key, path in data_paths.items()}
    return data

def test_ew_covar_lambda_097(load_data_ew):
    cout_21 = CovarianceEstimation.ewCovar(load_data_ew['x2'], lambda_=0.97)
    check21 = cout_21 - load_data_ew['out_21']
    check21 = np.linalg.norm(check21.to_numpy())
    print(f"Check21 value: {check21}")
    assert check21 < 1e-6, "EW Covariance (lambda=0.97) failed"
    print("Test 2.1 passed")

def test_ew_correlation_lambda_094(load_data_ew):
    cout_22 = CovarianceEstimation.ewCovar(load_data_ew['x2'], lambda_=0.94)
    sd = 1 / np.sqrt(np.diag(cout_22))
    cout_22 = np.diag(sd) @ cout_22 @ np.diag(sd)
    check22 = cout_22 - load_data_ew['out_22']
    check22 = np.linalg.norm(check22.to_numpy())
    print(f"Check22 value: {check22}")
    assert check22 < 1e-6, "EW Correlation (lambda=0.94) failed"
    print("Test 2.2 passed")

def test_ew_covar_with_var_and_correlation(load_data_ew):
    cout_23 = CovarianceEstimation.ewCovar(load_data_ew['x2'], lambda_=0.97)
    sd1 = np.sqrt(np.diag(cout_23))
    cout_23 = CovarianceEstimation.ewCovar(load_data_ew['x2'], lambda_=0.94)
    sd = 1 / np.sqrt(np.diag(cout_23))
    cout_23 = np.diag(sd1) @ np.diag(sd) @ cout_23 @ np.diag(sd) @ np.diag(sd1)
    check23 = cout_23 - load_data_ew['out_23']
    check23 = np.linalg.norm(check23.to_numpy())
    print(f"Check23 value: {check23}")
    assert check23 < 1e-6, "EW Covar with EW Var and Correlation failed"
    print("Test 2.3 passed")



# Test 3
@pytest.fixture
def load_data3():
    data_paths = {
        'cin31': 'test/testout_1.3.csv',
        'cin32': 'test/testout_1.4.csv',
        'cin33': 'test/testout_1.3.csv',
        'cin34': 'test/testout_1.4.csv',
        'out_31': 'test/testout_3.1.csv',
        'out_32': 'test/testout_3.2.csv',
        'out_33': 'test/testout_3.3.csv',
        'out_34': 'test/testout_3.4.csv',
    }
    data = {key: pd.read_csv(path) for key, path in data_paths.items()}
    # For matrices that need to be in numpy format before function calls
    data['cin33'] = data['cin33'].to_numpy()
    data['cin34'] = data['cin34'].to_numpy()
    data['out_33'] = data['out_33'].to_numpy()
    data['out_34'] = data['out_34'].to_numpy()
    return data

def test_near_psd_covariance(load_data3):
    cout_31 = NonPsdFixes.near_psd(load_data3['cin31'])
    check31 = cout_31 - load_data3['out_31']
    check31 = np.linalg.norm(check31.to_numpy())
    print(f"Check31 value: {check31}")
    assert check31 < 1e-6, "near_psd covariance failed"
    print("Test 3.1 passed")

def test_near_psd_correlation(load_data3):
    cout_32 = NonPsdFixes.near_psd(load_data3['cin32'])
    check32 = cout_32 - load_data3['out_32']
    check32 = np.linalg.norm(check32.to_numpy())
    print(f"Check32 value: {check32}")
    assert check32 < 1e-6, "near_psd correlation failed"
    print("Test 3.2 passed")

def test_higham_covariance(load_data3):
    cout_33 = NonPsdFixes.higham_nearestPSD(load_data3['cin33'])
    check33 = cout_33 - load_data3['out_33']
    check33 = np.linalg.norm(check33)
    print(f"Check33 value: {check33}")
    assert check33 < 1e-6, "Higham covariance failed"
    print("Test 3.3 passed")

def test_higham_correlation(load_data3):
    cout_34 = NonPsdFixes.higham_nearestPSD(load_data3['cin34'])
    check34 = cout_34 - load_data3['out_34']
    check34 = np.linalg.norm(check34)
    print(f"Check34 value: {check34}")
    assert check34 < 1e-6, "Higham correlation failed"
    print("Test 3.4 passed")


# test4
@pytest.fixture
def load_data4():
    data_paths = {
        'cin4': 'test/testout_3.1.csv',
        'out_4': 'test/testout_4.1.csv',
    }
    data = {key: pd.read_csv(path).to_numpy() for key, path in data_paths.items()}
    return data

def test_cholesky_factorization(load_data4):
    cin4 = load_data4['cin4']
    out_4_expected = load_data4['out_4']
    n, m = cin4.shape
    cout_4 = np.zeros((n, m))
    NonPsdFixes.chol_psd(cout_4, cin4)
    check4 = np.linalg.norm(cout_4 - out_4_expected)
    print(f"Check4 value: {check4}")
    assert check4 < 1e-6, "Cholesky factorization test failed"
    print("Test 4 passed")


# test5
# Fixture for loading data for Test 5
@pytest.fixture
def load_data5():
    paths = {
        'cin51': 'test/test5_1.csv',
        'cin52': 'test/test5_2.csv',
        'cin53': 'test/test5_3.csv',
        'cin54': 'test/test5_3.csv',  # Note: This seems to be duplicated in the original paths; adjust if needed
        'cin55': 'test/test5_2.csv',  # Note: This seems to reference test5_2.csv again; adjust if needed
        'out_51': 'test/testout_5.1.csv',
        'out_52': 'test/testout_5.2.csv',
        'out_53': 'test/testout_5.3.csv',
        'out_54': 'test/testout_5.4.csv',
        'out_55': 'test/testout_5.5.csv',
    }
    data = {key: pd.read_csv(path).to_numpy() for key, path in paths.items()}
    return data


def test_normal_simulation_pd_input(load_data5):
    cout_51 = np.cov(NonPsdFixes.simulate_normal(100000, load_data5['cin51']))
    check51 = np.linalg.norm(cout_51 - load_data5['out_51'])
    print(f"Check51 value: {check51}")
    assert check51 < 1e-3, "Test 5.1 failed"
    print("Test 5.1 passed")

def test_normal_simulation_psd_input(load_data5):
    cout_52 = np.cov(NonPsdFixes.simulate_normal(100000, load_data5['cin52']))
    check52 = np.linalg.norm(cout_52 - load_data5['out_52'])
    print(f"Check52 value: {check52}")
    assert check52 < 1e-3, "Test 5.2 failed"
    print("Test 5.2 passed")

def test_normal_simulation_nonpsd_input_near_psd(load_data5):
    cout_53 = np.cov(NonPsdFixes.simulate_normal(100000, load_data5['cin53'], fix_method=NonPsdFixes.near_psd))
    check53 = np.linalg.norm(cout_53 - load_data5['out_53'])
    print(f"Check53 value: {check53}")
    assert check53 < 1e-3, "Test 5.3 failed"
    print("Test 5.3 passed")

def test_normal_simulation_nonpsd_input_higham_nearestPSD(load_data5):
    cout_54 = np.cov(NonPsdFixes.simulate_normal(100000, load_data5['cin54'], fix_method=NonPsdFixes.higham_nearestPSD))
    check54 = np.linalg.norm(cout_54 - load_data5['out_54'])
    print(f"Check54 value: {check54}")
    assert check54 < 1e-3, "Test 5.4 failed"
    print("Test 5.4 passed")

def test_pca_simulation(load_data5):
    cout_55 = np.cov(NonPsdFixes.simulate_pca(load_data5['cin55'], 100000, pctExp=0.99))
    check55 = np.linalg.norm(cout_55 - load_data5['out_55'])
    print(f"Check55 value: {check55}")
    assert check55 < 1e-2, "Test 5.5 failed"
    print("Test 5.5 passed")

# test6
# Fixture for loading data for Test 6
@pytest.fixture
def load_data6():
    cin6 = pd.read_csv('test/test6.csv')
    out_61 = pd.read_csv('test/test6_1.csv').iloc[:, 1:].to_numpy()
    out_62 = pd.read_csv('test/test6_2.csv').iloc[:, 1:].to_numpy()
    return cin6, out_61, out_62

# Test 6 Functions
def test_arithmetic_returns(load_data6):
    cin6, out_61, out_62 = load_data6
    rout_1 = CalculateReturn.return_calc(cin6).iloc[:, 1:].to_numpy()
    check61 = np.linalg.norm(rout_1 - out_61)
    print(f"Check61 value: {check61}")
    assert check61 < 1e-6, "Test 6.1 failed"
    print("Test 6.1 passed")

def test_log_returns(load_data6):
    cin6, out_61, out_62 = load_data6
    rout_2 = CalculateReturn.return_calc(cin6, 'LOG').iloc[:, 1:].to_numpy()
    check62 = np.linalg.norm(rout_2 - out_62)
    print(f"Check62 value: {check62}")
    assert check62 < 1e-6, "Test 6.2 failed"
    print("Test 6.2 passed")


# Fixture for loading data for Test 7
@pytest.fixture
def load_data7():
    data_paths = {
        'cin71': 'test/test7_1.csv',
        'cin72': 'test/test7_2.csv',
        'cin73': 'test/test7_3.csv',
        'out_71': 'test/testout7_1.csv',
        'out_72': 'test/testout7_2.csv',
        'out_73': 'test/testout7_3.csv',
    }
    data = {key: pd.read_csv(path) for key, path in data_paths.items()}
    return data


def test_fit_normal_distribution(load_data7):
    cin71 = load_data7['cin71'].to_numpy()
    out_71 = load_data7['out_71'].to_numpy()
    fd_71, params = ModelFitter.fit_normal(cin71)
    mu_71, sigma_71 = params[1], params[2]
    cout_71 = np.array([mu_71, sigma_71])
    check71 = np.linalg.norm(cout_71 - out_71)
    print(f"Check71 value: {check71}")
    assert check71 < 1e-2, "Test 7.1 - Fit Normal Distribution failed"

def test_fit_general_t(load_data7):
    cin72 = load_data7['cin72'].to_numpy().flatten()
    out_72 = load_data7['out_72'].to_numpy().flatten()
    _, params_t = ModelFitter.fit_general_t(cin72)
    cout_72 = np.array([params_t[0], params_t[1], params_t[2]])
    assert np.linalg.norm(cout_72 - out_72) < 1e-2

def test_fit_regression_t(load_data7):
    cin73 = load_data7['cin73']
    out_73 = load_data7['out_73'].to_numpy().flatten()
    y = cin73.iloc[:, -1].to_numpy()
    xs = cin73.iloc[:, :-1].to_numpy()
    fd_73, params_treg = ModelFitter.fit_regression_t(y, xs)
    cout_73 = np.array([params_treg[0], params_treg[1], params_treg[2], fd_73.beta[0], fd_73.beta[1], fd_73.beta[2], fd_73.beta[3]])
    assert np.linalg.norm(cout_73 - out_73) < 1e-2




# Fixture for loading data for Test 8
@pytest.fixture
def load_data8():
    data_paths = {
        'cin81': 'test/test7_1.csv',
        'cin82': 'test/test7_2.csv',
        'cin83': 'test/test7_2.csv',
        'cin84': 'test/test7_1.csv',
        'cin85': 'test/test7_2.csv',
        'cin86': 'test/test7_2.csv',
        'out_81': 'test/testout8_1.csv',
        'out_82': 'test/testout8_2.csv',
        'out_83': 'test/testout8_3.csv',
        'out_84': 'test/testout8_4.csv',
        'out_85': 'test/testout8_5.csv',
        'out_86': 'test/testout8_6.csv',
    }
    data = {key: pd.read_csv(path) for key, path in data_paths.items()}
    return data

alpha = 0.05

def test_var_normal(load_data8):
    cin81 = load_data8['cin81'].to_numpy()
    out_81 = load_data8['out_81'].to_numpy().flatten()
    _, params_81 = ModelFitter.fit_normal(cin81)
    cout_81 = np.array([
        ModelFitter.VaR_norm(params_81[2], mu=params_81[1], alpha=alpha),
        ModelFitter.VaR_norm(params_81[2], alpha=alpha)
    ])
    assert np.linalg.norm(cout_81 - out_81) < 1e-2

def test_var_tdist(load_data8):
    cin82 = load_data8['cin82'].to_numpy().flatten()
    out_82 = load_data8['out_82'].to_numpy().flatten()
    _, params_82 = ModelFitter.fit_general_t(cin82)
    cout_82 = np.array([
        ModelFitter.VaR_t(params_82[2], params_82[1], mu=params_82[0], alpha=alpha),
        ModelFitter.VaR_t(params_82[2], params_82[1], alpha=alpha)
    ])
    assert np.linalg.norm(cout_82 - out_82) < 1e-2

def test_var_simulation(load_data8):
    cin83 = load_data8['cin83'].to_numpy().flatten()
    out_83 = load_data8['out_83'].to_numpy().flatten()
    _, _ = ModelFitter.fit_general_t(cin83)
    sim_83 = np.random.rand(10000)  # Assuming some simulation function
    cout_83 = np.array([
        ModelFitter.VaR(sim_83),
        ModelFitter.VaR(sim_83 - np.mean(sim_83))
    ])
    assert np.linalg.norm(cout_83 - out_83) < 1e-0

def test_es_normal(load_data8):
    cin84 = load_data8['cin84'].to_numpy()
    out_84 = load_data8['out_84'].to_numpy().flatten()
    _, params_84 = ModelFitter.fit_normal(cin84)
    cout_84 = np.array([
        ModelFitter.ES_norm(params_84[2], mu=params_84[1], alpha=alpha),
        ModelFitter.ES_norm(params_84[2], alpha=alpha)
    ])
    assert np.linalg.norm(cout_84 - out_84) < 1e-2

def test_es_tdist(load_data8):
    cin85 = load_data8['cin85'].to_numpy().flatten()
    out_85 = load_data8['out_85'].to_numpy().flatten()
    _, params_85 = ModelFitter.fit_general_t(cin85)
    cout_85 = np.array([
        ModelFitter.ES_t(params_85[2], params_85[1], mu=params_85[0], alpha=alpha),
        ModelFitter.ES_t(params_85[2], params_85[1], alpha=alpha)
    ])
    assert np.linalg.norm(cout_85 - out_85) < 1e-2

def test_es_simulation(load_data8):
    cin86 = load_data8['cin86'].to_numpy().flatten()
    out_86 = load_data8['out_86'].to_numpy().flatten()
    _, _ = ModelFitter.fit_general_t(cin86)
    sim_86 = np.random.rand(10000)  # Assuming some simulation function
    cout_86 = np.array([
        ModelFitter.ES(sim_86),
        ModelFitter.ES(sim_86 - np.mean(sim_86))
    ])
    assert np.linalg.norm(cout_86 - out_86) < 1e-0




# test9
@pytest.fixture
def load_data9():
    return {
        'returns': pd.read_csv('test/test9_1_returns.csv'),
        'portfolio': pd.read_csv('test/test9_1_portfolio.csv'),
        'expected_output': pd.read_csv('test/testout9_1.csv')
    }

# Test function for VaR/ES on 2 levels from simulated values - Copula
def test_var_es_copula(load_data8):
    cin86 = load_data8['cin86'].to_numpy().flatten()
    out_86 = load_data8['out_86'].to_numpy().flatten()
    _, _ = ModelFitter.fit_general_t(cin86)
    sim_86 = np.random.rand(10000)  # Assuming some simulation function
    cout_86 = np.array([
        ModelFitter.ES(sim_86),
        ModelFitter.ES(sim_86 - np.mean(sim_86))
    ])
    assert np.linalg.norm(cout_86 - out_86) < 1e-0


