import pandas as pd
import numpy as np
import streamlit as st

import plotly.graph_objects as go
from scipy.optimize import curve_fit, minimize

for k, v in st.session_state.items():
    st.session_state[k] = v

# make page widescreen
st.set_page_config(
    page_title="Media Optimiser",
    layout="wide",
    initial_sidebar_state="expanded")

tab1, tab2, tab3 = st.tabs(["Meta", "TTD", "Overall"])

with tab1:
    dfUAE = pd.read_csv("Data/UAE-Meta.csv")
    dfUAE["Market"] = "UAE"
    dfDE = pd.read_csv("Data/DE-Meta.csv")
    dfDE["Market"] = "DE"
    dfUK = pd.read_csv("C:/Users/maahaque/PycharmProjects/MediaOptimiserDemo/Data/UK-Meta.csv")
    dfUK["Market"] = "UK"

    date_format = "%b %d, %Y"

    df_list = [dfUAE, dfDE, dfUK]

    totalcurve_dict = {}
    df2 = pd.DataFrame()
    df_dict = {}
    container = st.container()
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = container.columns(9)
    with col1:
        budget = st.number_input("Set budget", min_value=200, value=1000, key='metabudget', step=500)


    def OptimiseBudgetsHill(budget, df_dict, df):

        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))

        # set budget to budget
        budget = budget

        def channel_optimizer(int_budget_list, channels):
            res_list = [
                hill(int_budget_list[i], df_dict[channels[i]][0], df_dict[channels[i]][1], df_dict[channels[i]][2]) for
                i in range(len(int_budget_list))]
            calculation = (sum(res_list))
            return -1 * calculation

        # define intitial budget where its spread equally
        def int_budget_maker(number_channels, budget):
            '''equaly weughted budget maker'''
            budget_share = budget / number_channels
            initial_budget = [budget * 0.36, budget * 0.25, budget * 0.39]
            return initial_budget

        # define constraints
        def equal_cons(x):
            '''must be a constraint equal to the budget'''
            x_list = []
            for i in range(len(x)):
                x_list.append(x[i])

            return sum(x_list) - budget

        # define bounds
        bounds = []
        for x in range(len(df)):
            bounds.append((0, budget))

        # define constraints
        constraint = {'type': 'eq'
            , 'fun': equal_cons
                      # ,'args': (len(to_budget_df),)
                      }
        constraint = [constraint]

        # call minimize function with all the above to get results
        result = minimize(channel_optimizer, int_budget_maker(len(df), budget), args=(list(df['Market'].unique())),
                          jac='3-point', hess=None, bounds=bounds, constraints=constraint)
        # make df_results dataframe with channel names and optimal ads per week
        df_results = pd.DataFrame(list(zip(df_dict.keys(), result.x)), columns=['Channel', 'Optimal Ads Per Week'])
        # round optimal ads per week and no decimal places
        # df_results['Optimal Ads Per Week'] = df_results['Optimal Ads Per Week'].round(decimals=0)
        # sort by highest optimal ads
        df_results = df_results.sort_values(by="Optimal Ads Per Week", ascending=False).reset_index(drop=True)
        # output as streamlit dataframe
        # st.dataframe(df_results.style.format(
        #     subset=["Optimal Ads Per Week"], formatter="{:.0f}"), use_container_width=True)
        return df_results


    dfcombined = pd.concat([dfUAE, dfUK, dfDE])

    optimaly = []
    marginalroi = []
    for df in df_list:
        df['Week'] = pd.to_datetime(df['Week'], format=date_format).dt.date
        df = df.sort_values(by='Week')
        market = df["Market"][0]


        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))


        df = df[df["Costs ($)"] > 100]
        # set xdata and ydata
        xdata = df["Costs ($)"].astype(float)
        ydata = df["Revenue"].astype(float)

        # run curve_fit function to get params in popt
        # popt, pcov = curve_fit(hill, xdata, ydata, maxfev=10000, p0=[2, np.median(xdata), max(ydata)])

        p0_list = [[2, np.median(xdata), max(ydata)], [2, np.median(ydata), max(ydata)], [1, max(xdata), max(ydata)]]
        param_bounds = [[1.5, 200, min(ydata)], [3, np.inf, max(ydata)]]


        popt, pcov = curve_fit(hill, xdata, ydata, p0=[2, np.median(xdata), max(ydata)], maxfev=5000, bounds=param_bounds)

        # Generate new x values for the curve
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # Evaluate the curve at the new x values using the fitted parameters
        y_curve = hill(x_curve, popt[0], popt[1], popt[2])

        curve_dict = {}
        curve_dict[market] = {'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]], 'xdata': xdata.tolist(),
                              'ydata': ydata.tolist(), 'x_curve': x_curve.tolist(), 'y_curve': y_curve.tolist()}
        df1 = pd.DataFrame({'Market': market, 'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]]})
        df2 = df2.append(df1)

        totalcurve_dict.update(curve_dict)
        channel_list = list(df2['Market'])
        alpha_list = list(df2['a'])
        beta_list = list(df2['b'])
        gamma_list = list(df2['c'])
        alpha_beta = list(zip(alpha_list, beta_list, gamma_list))

        # loop through and construct dictionary with parameters
        for idx, key in enumerate(channel_list):
            df_dict[key] = alpha_beta[idx]

    df_results = OptimiseBudgetsHill(budget, df_dict, df2)

    df_results = df_results.rename({'Channel': 'Market', 'Optimal Ads Per Week': "Costs ($)"}, axis=1)

    key_order = df_results["Market"].tolist()
    totalcurve_dict = {k: totalcurve_dict[k] for k in key_order if k in totalcurve_dict}

    for key, value in totalcurve_dict.items():

        def hill(x, S, K, beta):
            return beta - (K ** S * beta) / (x ** S + K ** S)


        # set medium to be the dictionary's key (which is the channel)
        market = key

        # set xdata and ydata as series
        xdata = pd.Series(value['xdata'])
        ydata = pd.Series(value['ydata'])

        # create a Plotly figure
        fig = go.Figure()

        # Update the layout to include the title and axis labels
        fig.update_layout(
            title=market,
            xaxis_title="Costs ($)",
            xaxis_title_standoff=40,
            yaxis_title="Revenue"
        )

        # find optimal point from the optimiser
        optimal_x = float(df_results[df_results['Market'] == key]["Costs ($)"])

        # range of x values to plot curve for, 30% greater than optimal value
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # if max of xdata is greater than optimal, make the max based on max xdata, else 10% greater than optimal
        if max(xdata) > optimal_x:
            x_curve = np.arange(0, max(xdata) * 1.3, 1)
        else:
            x_curve = np.arange(0, optimal_x * 1.1, 1)

        # plot y logarithmic/sqrt curve for these x values with parameters from the fit curve
        y_curve = hill(x_curve, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))

        # find the corresponding optimal y point for the optimal x to plot optimal point on curve
        optimal_y = float(np.interp(optimal_x, x_curve, y_curve))
        marginal_y = float(np.interp(optimal_x + 1000, x_curve, y_curve))
        marginal_roi = (marginal_y - optimal_y) / 1000
        # append each y to the empty list
        optimaly.append(optimal_y)
        marginalroi.append(marginal_roi)

        # Add the data to the figure - first is points, second is log curve
        fig.add_trace(go.Scatter(x=xdata, y=ydata, name='Data', mode='markers'))
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, name='Fitted Response Curve'))

        # Add the optimal point as a separate trace
        fig.add_trace(
            go.Scatter(x=[optimal_x], y=[optimal_y], name='Optimal Point', mode='markers', marker=dict(size=17)))

        # Calculate R squared
        # y_mean = (np.mean(ydata))
        # ss_res = np.sum((ydata - hill(xdata, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))) ** 2)
        # ss_tot = np.sum((ydata - y_mean) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        #
        # fig.add_annotation(text="R <sup>2</sup> : " + str(round(r_squared * 100, 1)) + "%",
        #                    xref="paper", yref="paper",
        #                    x=1, y=1.2, showarrow=False)
        fig.add_annotation(text="Optimal Ads : " + '{:,}'.format(int(optimal_x)),
                           xref="paper", yref="paper",
                           x=1, y=1.14, showarrow=False)
        fig.add_annotation(text="Uplift Expected : " + '{:,}'.format(int(optimal_y)),
                           xref="paper", yref="paper",
                           x=1, y=1.08, showarrow=False)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    df_y_results = pd.DataFrame({"Revenue": optimaly, 'Marginal ROI': marginalroi})
    merged_df = pd.merge(df_results, df_y_results, left_index=True, right_index=True)
    merged_df.columns = [
        'Optimal ' + col if col == "Costs ($)" else 'Expected ' + col + " ($)" if col == "Revenue" else col
        for col in merged_df.columns]
    merged_df["ROI"] = merged_df["Expected Revenue ($)"] / merged_df["Optimal Costs ($)"]
    merged_df = merged_df[["Market","Optimal Costs ($)","Expected Revenue ($)", "ROI", "Marginal ROI"]]
    # container.write(sum(merged_df["Optimal " + "Costs ($)"]))
    # container.write(sum(merged_df["Expected " + "Revenue"]))
    # container.write((sum(merged_df["Expected " + "Revenue"])-sum(merged_df["Optimal " + "Costs ($)"]))/sum(merged_df["Optimal " + "Costs ($)"]))

    with col4:
        st.metric("Total Budget", "${:,.0f}".format(merged_df["Optimal Costs ($)"].sum()))
    with col5:
        st.metric("Expected Revenue ($)", "${:,.0f}".format(merged_df["Expected Revenue ($)"].sum()))
    with col6:
        st.metric("ROI", "{:,.2f}".format(
            merged_df["Expected Revenue ($)"].sum() / merged_df["Optimal Costs ($)"].sum()))

    container.write("")
    container.dataframe(merged_df.style.format(subset=["Optimal Costs ($)", "Expected Revenue ($)"],
                                               formatter="{:,.0f}").format(subset=["ROI", "Marginal ROI"], formatter="{:,.1f}"),
                        use_container_width=True, height=(len(df_results) + 1) * 35 + 3)
with tab2:
    dfUAE = pd.read_csv("Data/UAE-TTD.csv")
    dfUAE["Market"] = "UAE"
    dfDE = pd.read_csv("DE-TTD.csv")
    dfDE["Market"] = "DE"
    dfUK = pd.read_csv("UK-TTD.csv")
    dfUK["Market"] = "UK"

    date_format = "%b %d, %Y"

    df_list = [dfUAE, dfDE, dfUK]

    totalcurve_dict = {}
    df2 = pd.DataFrame()
    df_dict = {}
    container = st.container()
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = container.columns(9)
    with col1:
        budget = st.number_input("Set budget", min_value=200, value=1000, key='ttdbudget', step=500)


    def OptimiseBudgetsHill(budget, df_dict, df):

        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))

        # set budget to budget
        budget = budget

        def channel_optimizer(int_budget_list, channels):
            res_list = [
                hill(int_budget_list[i], df_dict[channels[i]][0], df_dict[channels[i]][1], df_dict[channels[i]][2]) for
                i in range(len(int_budget_list))]
            calculation = (sum(res_list))
            return -1 * calculation

        # define intitial budget where its spread equally
        def int_budget_maker(number_channels, budget):
            '''equaly weughted budget maker'''
            budget_share = budget / number_channels
            initial_budget = [budget * 0.20, budget * 0.20, budget * 0.60]
            return initial_budget

        # define constraints
        def equal_cons(x):
            '''must be a constraint equal to the budget'''
            x_list = []
            for i in range(len(x)):
                x_list.append(x[i])

            return sum(x_list) - budget

        # define bounds
        bounds = []
        for x in range(len(df)):
            bounds.append((0, budget))

        # define constraints
        constraint = {'type': 'eq'
            , 'fun': equal_cons
                      # ,'args': (len(to_budget_df),)
                      }
        constraint = [constraint]

        # call minimize function with all the above to get results
        result = minimize(channel_optimizer, int_budget_maker(len(df), budget), args=(list(df['Market'].unique())),
                          jac='3-point', hess=None, bounds=bounds, constraints=constraint)
        # make df_results dataframe with channel names and optimal ads per week
        df_results = pd.DataFrame(list(zip(df_dict.keys(), result.x)), columns=['Channel', 'Optimal Ads Per Week'])
        # round optimal ads per week and no decimal places
        # df_results['Optimal Ads Per Week'] = df_results['Optimal Ads Per Week'].round(decimals=0)
        # sort by highest optimal ads
        df_results = df_results.sort_values(by="Optimal Ads Per Week", ascending=False).reset_index(drop=True)
        # output as streamlit dataframe
        # st.dataframe(df_results.style.format(
        #     subset=["Optimal Ads Per Week"], formatter="{:.0f}"), use_container_width=True)
        return df_results


    dfcombined = pd.concat([dfUAE, dfUK, dfDE])

    optimaly = []
    marginalroi = []

    for df in df_list:
        df['Week'] = pd.to_datetime(df['Week'], format=date_format).dt.date
        df = df.sort_values(by='Week')
        market = df["Market"][0]


        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))


        df = df[df["Costs ($)"] > 100]
        # set xdata and ydata
        xdata = df["Costs ($)"].astype(float)
        ydata = df["Revenue"].astype(float)

        # run curve_fit function to get params in popt
        # popt, pcov = curve_fit(hill, xdata, ydata, maxfev=10000, p0=[2, np.median(xdata), max(ydata)])

        p0_list = [[2, np.median(xdata), max(ydata)], [2, np.median(ydata), max(ydata)], [1, max(xdata), max(ydata)]]
        param_bounds = [[1.5, 200, min(ydata)], [3, np.inf, max(ydata)]]


        popt, pcov = curve_fit(hill, xdata, ydata, p0=[2, np.median(xdata), max(ydata)], maxfev=5000, bounds=param_bounds)

        # Generate new x values for the curve
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # Evaluate the curve at the new x values using the fitted parameters
        y_curve = hill(x_curve, popt[0], popt[1], popt[2])

        curve_dict = {}
        curve_dict[market] = {'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]], 'xdata': xdata.tolist(),
                              'ydata': ydata.tolist(), 'x_curve': x_curve.tolist(), 'y_curve': y_curve.tolist()}
        df1 = pd.DataFrame({'Market': market, 'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]]})
        df2 = df2.append(df1)

        totalcurve_dict.update(curve_dict)
        channel_list = list(df2['Market'])
        alpha_list = list(df2['a'])
        beta_list = list(df2['b'])
        gamma_list = list(df2['c'])
        alpha_beta = list(zip(alpha_list, beta_list, gamma_list))

        # loop through and construct dictionary with parameters
        for idx, key in enumerate(channel_list):
            df_dict[key] = alpha_beta[idx]

    df_results = OptimiseBudgetsHill(budget, df_dict, df2)

    df_results = df_results.rename({'Channel': 'Market', 'Optimal Ads Per Week': "Costs ($)"}, axis=1)

    key_order = df_results["Market"].tolist()
    totalcurve_dict = {k: totalcurve_dict[k] for k in key_order if k in totalcurve_dict}

    for key, value in totalcurve_dict.items():

        def hill(x, S, K, beta):
            return beta - (K ** S * beta) / (x ** S + K ** S)


        # set medium to be the dictionary's key (which is the channel)
        market = key

        # set xdata and ydata as series
        xdata = pd.Series(value['xdata'])
        ydata = pd.Series(value['ydata'])

        # create a Plotly figure
        fig = go.Figure()

        # Update the layout to include the title and axis labels
        fig.update_layout(
            title=market,
            xaxis_title="Costs ($)",
            xaxis_title_standoff=40,
            yaxis_title="Revenue"
        )

        # find optimal point from the optimiser
        optimal_x = float(df_results[df_results['Market'] == key]["Costs ($)"])

        # range of x values to plot curve for, 30% greater than optimal value
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # if max of xdata is greater than optimal, make the max based on max xdata, else 10% greater than optimal
        if max(xdata) > optimal_x:
            x_curve = np.arange(0, max(xdata) * 1.3, 1)
        else:
            x_curve = np.arange(0, optimal_x * 1.1, 1)

        # plot y logarithmic/sqrt curve for these x values with parameters from the fit curve
        y_curve = hill(x_curve, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))

        # find the corresponding optimal y point for the optimal x to plot optimal point on curve
        optimal_y = float(np.interp(optimal_x, x_curve, y_curve))
        marginal_y = float(np.interp(optimal_x + 1000, x_curve, y_curve))
        marginal_roi = (marginal_y - optimal_y) / 1000
        # append each y to the empty list
        optimaly.append(optimal_y)
        marginalroi.append(marginal_roi)

        # Add the data to the figure - first is points, second is log curve
        fig.add_trace(go.Scatter(x=xdata, y=ydata, name='Data', mode='markers'))
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, name='Fitted Response Curve'))

        # Add the optimal point as a separate trace
        fig.add_trace(
            go.Scatter(x=[optimal_x], y=[optimal_y], name='Optimal Point', mode='markers', marker=dict(size=17)))

        # # Calculate R squared
        # y_mean = (np.mean(ydata))
        # ss_res = np.sum(
        #     (ydata - hill(xdata, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))) ** 2)
        # ss_tot = np.sum((ydata - y_mean) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        #
        # fig.add_annotation(text="R <sup>2</sup> : " + str(round(r_squared * 100, 1)) + "%",
        #                    xref="paper", yref="paper",
        #                    x=1, y=1.2, showarrow=False)
        fig.add_annotation(text="Optimal Ads : " + '{:,}'.format(int(optimal_x)),
                           xref="paper", yref="paper",
                           x=1, y=1.14, showarrow=False)
        fig.add_annotation(text="Uplift Expected : " + '{:,}'.format(int(optimal_y)),
                           xref="paper", yref="paper",
                           x=1, y=1.08, showarrow=False)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    df_y_results = pd.DataFrame({"Revenue": optimaly, 'Marginal ROI': marginalroi})
    merged_df = pd.merge(df_results, df_y_results, left_index=True, right_index=True)
    merged_df.columns = [
        'Optimal ' + col if col == "Costs ($)" else 'Expected ' + col + " ($)" if col == "Revenue" else col
        for col in merged_df.columns]
    merged_df["ROI"] = merged_df["Expected Revenue ($)"] / merged_df["Optimal Costs ($)"]
    merged_df = merged_df[["Market","Optimal Costs ($)","Expected Revenue ($)", "ROI", "Marginal ROI"]]


    with col4:
        st.metric("Total Budget", "${:,.0f}".format(merged_df["Optimal Costs ($)"].sum()))
    with col5:
        st.metric("Expected Revenue ($)", "${:,.0f}".format(merged_df["Expected Revenue ($)"].sum()))
    with col6:
        st.metric("ROI", "{:,.2f}".format(
            merged_df["Expected Revenue ($)"].sum() / merged_df["Optimal Costs ($)"].sum()))

    container.write("")
    container.dataframe(merged_df.style.format(subset=["Optimal Costs ($)", "Expected Revenue ($)"],
                                               formatter="{:,.0f}").format(subset=["ROI", "Marginal ROI"], formatter="{:,.1f}"),
                        use_container_width=True, height=(len(df_results) + 1) * 35 + 3)

with tab3:
    dfUAEmeta = pd.read_csv("Data/UAE-Meta.csv")
    dfUAEmeta["Market"] = "META-UAE"
    dfDEmeta = pd.read_csv("Data/DE-Meta.csv")
    dfDEmeta["Market"] = "META-DE"
    dfUKmeta = pd.read_csv("Data/UK-Meta.csv")
    dfUKmeta["Market"] = "META-UK"
    dfUAEttd = pd.read_csv("Data/UAE-TTD.csv")
    dfUAEttd["Market"] = "TTD-UAE"
    dfDEttd = pd.read_csv("Data/DE-TTD.csv")
    dfDEttd["Market"] = "TTD-DE"
    dfUKttd = pd.read_csv("Data/UK-TTD.csv")
    dfUKttd["Market"] = "TTD-UK"

    st.write("")

    date_format = "%b %d, %Y"

    df_list = [dfUAEmeta, dfDEmeta, dfUKmeta, dfUAEttd, dfDEttd, dfUKttd]

    totalcurve_dict = {}
    df2 = pd.DataFrame()
    df_dict = {}

    container = st.container()
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = container.columns(9)
    with col1:
        budget = st.number_input("Set budget", min_value=200, value=1000, key='overallbudget', step=500)


    def OptimiseBudgetsHill(budget, df_dict, df):

        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))

        # set budget to budget
        budget = budget

        def channel_optimizer(int_budget_list, channels):
            res_list = [
                hill(int_budget_list[i], df_dict[channels[i]][0], df_dict[channels[i]][1], df_dict[channels[i]][2]) for
                i in range(len(int_budget_list))]
            calculation = (sum(res_list))
            return -1 * calculation

        # define intitial budget where its spread equally
        def int_budget_maker(number_channels, budget):
            '''equaly weughted budget maker'''
            budget_share = budget / number_channels
            initial_budget = [budget * 0.26, budget * 0.18, budget * 0.29, budget * 0.05, budget * 0.05, budget * 0.15]
            return initial_budget

        # define constraints
        def equal_cons(x):
            '''must be a constraint equal to the budget'''
            x_list = []
            for i in range(len(x)):
                x_list.append(x[i])

            return sum(x_list) - budget

        # define bounds
        bounds = []
        for x in range(len(df)):
            bounds.append((0, budget))

        # define constraints
        constraint = {'type': 'eq'
            , 'fun': equal_cons
                      # ,'args': (len(to_budget_df),)
                      }
        constraint = [constraint]

        # call minimize function with all the above to get results
        result = minimize(channel_optimizer, int_budget_maker(len(df), budget), args=(list(df['Market'].unique())),
                          jac='3-point', hess=None, bounds=bounds, constraints=constraint)
        # make df_results dataframe with channel names and optimal ads per week
        df_results = pd.DataFrame(list(zip(df_dict.keys(), result.x)), columns=['Channel', 'Optimal Ads Per Week'])
        # round optimal ads per week and no decimal places
        # df_results['Optimal Ads Per Week'] = df_results['Optimal Ads Per Week'].round(decimals=0)
        # sort by highest optimal ads
        df_results = df_results.sort_values(by="Optimal Ads Per Week", ascending=False).reset_index(drop=True)
        # output as streamlit dataframe
        # st.dataframe(df_results.style.format(
        #     subset=["Optimal Ads Per Week"], formatter="{:.0f}"), use_container_width=True)
        return df_results


    dfcombined = pd.concat([dfUAEmeta, dfDEmeta, dfUKmeta, dfUAEttd, dfDEttd, dfUKttd])

    optimaly = []
    marginalroi = []
    for df in df_list:
        df['Week'] = pd.to_datetime(df['Week'], format=date_format).dt.date
        df = df.sort_values(by='Week')
        market = df["Market"][0]


        def hill(x, S, K, beta):
            return beta - ((K ** S) * beta) / ((x ** S) + (K ** S))


        df = df[df["Costs ($)"] > 100]
        # set xdata and ydata
        xdata = df["Costs ($)"].astype(float)
        ydata = df["Revenue"].astype(float)

        # run curve_fit function to get params in popt
        # popt, pcov = curve_fit(hill, xdata, ydata, maxfev=10000, p0=[2, np.median(xdata), max(ydata)])

        p0_list = [[2, np.median(xdata), max(ydata)], [2, np.median(ydata), max(ydata)], [1, max(xdata), max(ydata)]]
        param_bounds = [[1.5, 200, min(ydata)], [3, np.inf, max(ydata)]]


        popt, pcov = curve_fit(hill, xdata, ydata, p0=[2, np.median(xdata), max(ydata)], maxfev=5000,
                               bounds=param_bounds)

        # Generate new x values for the curve
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # Evaluate the curve at the new x values using the fitted parameters
        y_curve = hill(x_curve, popt[0], popt[1], popt[2])

        curve_dict = {}
        curve_dict[market] = {'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]], 'xdata': xdata.tolist(),
                              'ydata': ydata.tolist(), 'x_curve': x_curve.tolist(), 'y_curve': y_curve.tolist()}
        df1 = pd.DataFrame({'Market': market, 'a': [popt[0]], 'b': [popt[1]], 'c': [popt[2]]})
        df2 = df2.append(df1)

        totalcurve_dict.update(curve_dict)
        channel_list = list(df2['Market'])
        alpha_list = list(df2['a'])
        beta_list = list(df2['b'])
        gamma_list = list(df2['c'])
        alpha_beta = list(zip(alpha_list, beta_list, gamma_list))

        # loop through and construct dictionary with parameters
        for idx, key in enumerate(channel_list):
            df_dict[key] = alpha_beta[idx]

    df_results = OptimiseBudgetsHill(budget, df_dict, df2)

    df_results = df_results.rename({'Channel': 'Market', 'Optimal Ads Per Week': "Costs ($)"}, axis=1)

    key_order = df_results["Market"].tolist()
    totalcurve_dict = {k: totalcurve_dict[k] for k in key_order if k in totalcurve_dict}

    for key, value in totalcurve_dict.items():

        def hill(x, S, K, beta):
            return beta - (K ** S * beta) / (x ** S + K ** S)


        # set medium to be the dictionary's key (which is the channel)
        market = key

        # set xdata and ydata as series
        xdata = pd.Series(value['xdata'])
        ydata = pd.Series(value['ydata'])

        # create a Plotly figure
        fig = go.Figure()

        # Update the layout to include the title and axis labels
        fig.update_layout(
            title=market,
            xaxis_title="Costs ($)",
            xaxis_title_standoff=40,
            yaxis_title="Revenue"
        )

        # find optimal point from the optimiser
        optimal_x = float(df_results[df_results['Market'] == key]["Costs ($)"])

        # range of x values to plot curve for, 30% greater than optimal value
        x_curve = np.arange(0, max(xdata) * 1.3, 1)

        # if max of xdata is greater than optimal, make the max based on max xdata, else 10% greater than optimal
        if max(xdata) > optimal_x:
            x_curve = np.arange(0, max(xdata) * 1.3, 1)
        else:
            x_curve = np.arange(0, optimal_x * 1.1, 1)

        # plot y logarithmic/sqrt curve for these x values with parameters from the fit curve
        y_curve = hill(x_curve, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))

        # find the corresponding optimal y point for the optimal x to plot optimal point on curve
        optimal_y = float(np.interp(optimal_x, x_curve, y_curve))
        marginal_y = float(np.interp(optimal_x + 1000, x_curve, y_curve))
        marginal_roi = (marginal_y - optimal_y) / 1000
        # append each y to the empty list
        optimaly.append(optimal_y)
        marginalroi.append(marginal_roi)
        # Add the data to the figure - first is points, second is log curve
        fig.add_trace(go.Scatter(x=xdata, y=ydata, name='Data', mode='markers'))
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, name='Fitted Response Curve'))

        # Add the optimal point as a separate trace
        fig.add_trace(
            go.Scatter(x=[optimal_x], y=[optimal_y], name='Optimal Point', mode='markers', marker=dict(size=17)))

        # Calculate R squared
        # y_mean = (np.mean(ydata))
        # ss_res = np.sum(
        #     (ydata - hill(xdata, float(value['a'][0]), float(value['b'][0]), float(value['c'][0]))) ** 2)
        # ss_tot = np.sum((ydata - y_mean) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        #
        # fig.add_annotation(text="R <sup>2</sup> : " + str(round(r_squared * 100, 1)) + "%",
        #                    xref="paper", yref="paper",
        #                    x=1, y=1.2, showarrow=False)
        fig.add_annotation(text="Optimal Ads : " + '{:,}'.format(int(optimal_x)),
                           xref="paper", yref="paper",
                           x=1, y=1.14, showarrow=False)
        fig.add_annotation(text="Uplift Expected : " + '{:,}'.format(int(optimal_y)),
                           xref="paper", yref="paper",
                           x=1, y=1.08, showarrow=False)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    df_y_results = pd.DataFrame({"Revenue": optimaly, 'Marginal ROI': marginalroi})
    merged_df = pd.merge(df_results, df_y_results, left_index=True, right_index=True)
    merged_df.columns = [
        'Optimal ' + col if col == "Costs ($)" else 'Expected ' + col + " ($)" if col == "Revenue" else col
        for col in merged_df.columns]
    merged_df["ROI"] = merged_df["Expected Revenue ($)"] / merged_df["Optimal Costs ($)"]
    merged_df = merged_df[["Market","Optimal Costs ($)","Expected Revenue ($)", "ROI", "Marginal ROI"]]


    with col4:
        st.metric("Total Budget", "${:,.0f}".format(merged_df["Optimal Costs ($)"].sum()))
    with col5:
        st.metric("Expected Revenue ($)", "${:,.0f}".format(merged_df["Expected Revenue ($)"].sum()))
    with col6:
        st.metric("ROI", "{:,.2f}".format(
            merged_df["Expected Revenue ($)"].sum() / merged_df["Optimal Costs ($)"].sum()))

    container.write("")
    container.dataframe(merged_df.style.format(subset=["Optimal Costs ($)", "Expected Revenue ($)"],
                                               formatter="{:,.0f}").format(subset=["ROI", "Marginal ROI"], formatter="{:,.1f}"),
                        use_container_width=True, height=(len(df_results) + 1) * 35 + 3)