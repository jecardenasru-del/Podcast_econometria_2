#funciones auxiliares para diagnostico de modelos de panel (EF, ER, Pooled) 

def pvar_summary(df, entity_col, time_col):
    novar_time, novar_ind = [], []
    for col in df.columns:
        if col in (entity_col, time_col):
            continue
        try:
            var_in_entity = df.groupby(entity_col)[col].nunique(dropna=False).max()
            var_in_time = df.groupby(time_col)[col].nunique(dropna=False).max()
        except TypeError:
            continue
        if var_in_entity <= 1:
            novar_time.append(col)
        if var_in_time <= 1:
            novar_ind.append(col)
    return novar_time, novar_ind

def pdim_summary(df, entity_col, time_col):
    n = df[entity_col].nunique()
    t_counts = df.groupby(entity_col)[time_col].nunique()
    balanceado = t_counts.nunique() == 1
    T = t_counts.iloc[0] if balanceado else t_counts.mean()
    N = len(df)
    return {"n_individuos": n, "T_periodos": T, "N_obs": N, "balanceado": balanceado}

def hausman_test(fe_result, re_result):
    b_fe, b_re = fe_result.params, re_result.params
    common = b_fe.index.intersection(b_re.index).drop("const", errors="ignore")
    b_diff = (b_fe[common] - b_re[common]).values
    v_diff = (fe_result.cov.loc[common, common] - re_result.cov.loc[common, common]).values
    stat = float(b_diff.T @ np.linalg.inv(v_diff) @ b_diff)
    df = len(common)
    pval = 1 - stats.chi2.cdf(stat, df)
    return stat, df, pval

def plm_lm_test(pooled_result, effect="individual"):
    e = pooled_result.resids
    idx_names = e.index.names

    def _lm_one_way(resid, group_level, other_size):
        g = resid.groupby(level=group_level)
        n_groups = g.ngroups
        Tbar = other_size
        sum_e2 = (resid ** 2).sum()
        sum_group_sq = (g.sum() ** 2).sum()
        lm = (n_groups * Tbar / (2 * (Tbar - 1))) * ((sum_group_sq / sum_e2) - 1) ** 2
        return lm

    N = e.index.get_level_values(idx_names[0]).nunique()
    T = e.index.get_level_values(idx_names[1]).nunique()

    if effect == "individual":
        lm = _lm_one_way(e, idx_names[0], T); df = 1
    elif effect == "time":
        lm = _lm_one_way(e, idx_names[1], N); df = 1
    elif effect == "twoways":
        lm = _lm_one_way(e, idx_names[0], T) + _lm_one_way(e, idx_names[1], N); df = 2
    else:
        raise ValueError("effect debe ser 'individual', 'time' o 'twoways'")
    pval = 1 - stats.chi2.cdf(lm, df)
    return lm, df, pval

def wooldridge_serial_test(fd_result, h0="fe"):
    """H0: no hay autocorrelacion serial en los errores idiosincraticos del
    modelo de niveles. Se aplica sobre los residuos de Primeras Diferencias:
    si el modelo de EF es correcto, Corr(e_it_FD, e_i,t-1_FD) = -0.5.
    """
    e = fd_result.resids
    d = e.reset_index()
    d.columns = ["entity", "time", "e"]
    d = d.sort_values(["entity", "time"])
    d["e_lag"] = d.groupby("entity")["e"].shift(1)
    d = d.dropna()
    import statsmodels.formula.api as smf
    mod = smf.ols("e ~ e_lag - 1", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["entity"]}
    )
    coef = mod.params["e_lag"]; se = mod.bse["e_lag"]
    target = -0.5 if h0 == "fe" else 0.0
    t_stat = (coef - target) / se
    df_resid = mod.df_resid
    pval = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
    return coef, se, t_stat, pval

def pesaran_cd_test(resid_series):
    """H0: no hay dependencia transversal (los residuos de distintos estados
    no estan correlacionados entre si en un mismo periodo)."""
    d = resid_series.reset_index()
    d.columns = ["entity", "time", "e"]
    wide = d.pivot(index="time", columns="entity", values="e")
    corr = wide.corr()
    N = corr.shape[0]; T = wide.shape[0]
    iu = np.triu_indices(N, k=1)
    rho_sum = corr.values[iu].sum()
    cd_stat = np.sqrt(2 * T / (N * (N - 1))) * rho_sum
    pval = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
    return cd_stat, pval

# ---- Funciones NUEVAS de diagnostico (no estaban en Guns.py) --------------
def jarque_bera_normalidad(resid):
    """H0: los residuos se distribuyen normal (asimetria=0, curtosis=3)."""
    jb_stat, jb_p, skew, kurt = jarque_bera(np.asarray(resid))
    return jb_stat, jb_p, skew, kurt

def bp_white_heterocedasticidad(resid, X):
    """Breusch-Pagan y White. H0: varianza del error constante (homocedasticidad).
    Se corre sobre los residuos del modelo de EF, usando como regresores las
    variables explicativas 'within' (X ya centrada implicitamente por PanelOLS)."""
    X_const = sm.add_constant(X)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, X_const)
    w_lm, w_lm_p, w_f, w_f_p = het_white(resid, X_const)
    return {
        "BP_LM": bp_lm, "BP_LM_p": bp_lm_p,
        "White_LM": w_lm, "White_LM_p": w_lm_p,
    }

def wald_modificado_heterocedasticidad_grupal(fe_result, entity_name="state"):
    """Test de Wald modificado (Greene 2000 / analogo a 'xttest3' de Stata).
    H0: la varianza del error es igual entre todos los estados (homocedasticidad
    'entre grupos'). Util cuando se sospecha que unos estados son mas
    'ruidosos' que otros (p. ej. estados pequenos con conteos de crimen bajos)."""
    e = fe_result.resids.reset_index()
    e.columns = [entity_name, "time", "e"]
    sigma2_i = e.groupby(entity_name)["e"].apply(lambda x: (x ** 2).mean())
    Ti = e.groupby(entity_name)["e"].count()
    sigma2 = (e["e"] ** 2).mean()
    stat = float(((sigma2_i - sigma2) ** 2 / (2 * sigma2 ** 2 / Ti)).sum())
    df = len(sigma2_i)
    pval = 1 - stats.chi2.cdf(stat, df)
    return stat, df, pval