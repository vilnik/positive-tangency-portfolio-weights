EVAL_PORTFOLIO_SPECS = {"value_weighted_monthly_k10_tc3":{"portfolio_size":10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency":"monthly",
                                                "weights_spec": "value_weighted",
                                                "rolling_window_days": 252*5,
                                                "display_name": "Value-Weighted"},
                        "value_weighted_monthly_k20_tc3":{"portfolio_size":20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency":"monthly",
                                                "weights_spec": "value_weighted",
                                                "rolling_window_days": 252*5,
                                                "display_name": "Value-Weighted"},
                        "equally_weighted_monthly_k10_tc3":{"portfolio_size":10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency":"monthly",
                                                "weights_spec": "equally_weighted",
                                                "rolling_window_days": 252*5,
                                                "display_name": "Equally-Weighted"},
                        "equally_weighted_monthly_k20_tc3": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "equally_weighted",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Equally-Weighted"},
                        "log_normal_monthly_k10_tc3_scale1": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "log_normal",
                                                "risk_aversion": 1.1,
                                                "scale": 1,
                                                "posterior_nu_plot_date": "2023-06-01",
                                                "rolling_window_days": 252*5,
                                                "prior_weights": "value_weighted",
                                                "display_name": "Log-Normal kappa=1"},
                        "log_normal_monthly_k20_tc3_scale1": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "log_normal",
                                                "risk_aversion": 1.06,
                                                "scale": 1,
                                                "posterior_nu_plot_date": "2023-06-01",
                                                "rolling_window_days": 252 * 5,
                                                "prior_weights": "value_weighted",
                                                "display_name": "Log-Normal kappa=1"},
                        "log_normal_monthly_k10_tc3_scale2": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "log_normal",
                                                "risk_aversion": 1.25,
                                                "scale": 2,
                                                "posterior_nu_plot_date": "2023-06-01",
                                                "rolling_window_days": 252*5,
                                                "prior_weights": "value_weighted",
                                                "display_name": "Log-Normal kappa=2"},
                        "log_normal_monthly_k20_tc3_scale2": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "log_normal",
                                                "risk_aversion": 1.15,
                                                "scale": 2,
                                                "posterior_nu_plot_date": "2023-06-01",
                                                "rolling_window_days": 252 * 5,
                                                "prior_weights": "value_weighted",
                                                "display_name": "Log-Normal kappa=2"},
                        "black_litterman_monthly_k10_tc3": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "black_litterman",
                                                "risk_aversion": 1.4,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252*5,
                                                "display_name": "Black-Litterman"},
                        "black_litterman_monthly_k20_tc3": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "black_litterman",
                                                "risk_aversion": 1.3,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Black-Litterman"},
                        "jorion_hyper_monthly_k10_tc3": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "jorion_hyper",
                                                "risk_aversion": 1.9,
                                                "covariance_matrix_estimation": "jorion_hyper",
                                                "rolling_window_days": 252*5,
                                                "display_name": "Jorion Hyperpar."},
                        "jorion_hyper_monthly_k20_tc3": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "jorion_hyper",
                                                "risk_aversion": 2.2,
                                                "covariance_matrix_estimation": "jorion_hyper",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Jorion Hyperpar."},
                        "shrinkage_monthly_k10_tc3": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "shrinkage",
                                                "risk_aversion": 9,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Shrinkage"},
                        "shrinkage_monthly_k20_tc3": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "shrinkage",
                                                "risk_aversion": 8,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Shrinkage"},
                        "min_variance_monthly_k10_tc3": {"portfolio_size": 10,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "min_variance",
                                                "risk_aversion": 9,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Min. Variance"},
                        "min_variance_monthly_k20_tc3": {"portfolio_size": 20,
                                                "turnover_cost_bps": 3,
                                                "rebalancing_frequency": "monthly",
                                                "weights_spec": "min_variance",
                                                "risk_aversion": 12,
                                                "covariance_matrix_estimation": "shrinkage",
                                                "rolling_window_days": 252 * 5,
                                                "display_name": "Min. Variance"}
}