import pytse_client as tse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from bidi.algorithm import get_display
import arabic_reshaper
import pyomo.environ as pyo


def read_from_csv(stock_symbols: list, directory_name: str = 'tickers_data'):
    stocks_data = {}
    for symbol in stock_symbols:
        file_path = os.path.join(directory_name, f"{symbol}.csv")

        if os.path.isfile(file_path):
            stocks_data[symbol] = pd.read_csv(file_path)
        else:
            print(f"File not found for {symbol}")
    return stocks_data


def calculate_monthly(stock_data, returns=True):
    monthly_returns = {}
    for symbol, df in stock_data.items():
        monthly_data = df.resample('M').agg({'open': 'first', 'close': 'last'})
        monthly_data.index = monthly_data.index.to_period('M')
        if returns:
            monthly_data['monthly_return'] = (monthly_data['close'] - monthly_data['open']) / monthly_data['open']
            monthly_returns[symbol] = monthly_data['monthly_return']
        else:
            monthly_data['close_price'] = monthly_data['close']
            monthly_returns[symbol] = monthly_data['close_price']
    return monthly_returns


def aggregated_raw_return_over_time(stock_list, monthly_returns):
    aggregated_returns = {}

    for month in monthly_returns[next(iter(monthly_returns))].index:
        aggregated_return = 0
        c = 0
        for stock in stock_list:
            if stock in monthly_returns and month in monthly_returns[stock].index:
                aggregated_return += monthly_returns[stock].loc[month]
                c += 1
        aggregated_returns[month] = aggregated_return / c

    return aggregated_returns


def trading_model(data, leaders, followers, initial_budget):
    model = pyo.ConcreteModel()

    monthly_returns = calculate_monthly(data)

    model.months = pyo.Set(initialize=monthly_returns[list(data.keys())[0]].index)
    model.stocks = pyo.Set(initialize=leaders + followers)

    model.profits = pyo.Var(model.months)
    model.holdings = pyo.Var(model.months, domain=pyo.NonNegativeReals, initialize=0)
    model.inventory_percent = pyo.Var(model.months, domain=pyo.NonNegativeReals, bounds=(0, 1), )

    leaders_aggregated_returns = aggregated_raw_return_over_time(leaders, monthly_returns)
    followers_aggregated_returns = aggregated_raw_return_over_time(followers, monthly_returns)

    # Initial budget
    model.holdings[model.months.first()] = initial_budget

    # Strategy constraint
    def strategy_rule(model, month):
        if month == model.months.first():
            return pyo.Constraint.Skip
        prev_month = model.months[model.months.ord(month) - 1]
        if leaders_aggregated_returns[prev_month] > 0:
            # Sell leaders, buy followers
            value = followers_aggregated_returns[month] - leaders_aggregated_returns[month]
        else:
            # Sell followers, buy leaders
            value = leaders_aggregated_returns[month] - followers_aggregated_returns[month]
        if not np.isnan(value):
            value = float(value)
        else:
            value = 0
        if value > 0:
            model.inventory_percent[month] = 0.50
        else:
            model.inventory_percent[month] = 0.25
        model.holdings[month] = model.holdings[prev_month] * (1 + (model.inventory_percent[month]) * value)
        return model.holdings[month] >= model.holdings[prev_month]

    model.strategy_constraint = pyo.Constraint(model.months, rule=strategy_rule)

    # Objective Function: Maximize holdings at the last month
    model.objective = pyo.Objective(expr=model.holdings[model.months.last()], sense=pyo.maximize)

    # Solve the model
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model)

    if (results.solver.status == pyo.SolverStatus.ok) and (
            results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("Solution is optimal.")
    elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible.")
    else:
        # Something else is wrong
        print("Solver Status: ", results.solver.status)
    months = []
    for month in model.months:
        if month != model.months.first():
            print(
                f"Month: {month}, Holdings: {pyo.value(model.holdings[month])}, Total asset Percent: {pyo.value(model.inventory_percent[month])}")
            months.append(month)
    return {'months': [pd.Period(month).to_timestamp() for month in months],
            'holdings': [pyo.value(model.holdings[month]) for month in months], }


def plot_profit(months, holdings):
    plt.figure(figsize=(10, 6))
    plt.plot(months, holdings, marker='o')
    plt.title('Holdings Over Time')
    plt.xlabel('Month')
    plt.ylabel('Holdings Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def prepare_data(stock_data):
    # find the stock with max length
    max_length = 0
    max_stock = None
    for s, d in stock_data.items():
        if len(d) > max_length:
            max_length = len(d)
            max_stock = s

    for s, d in stock_data.items():
        d['date'] = pd.to_datetime(d['date'])
        d.set_index('date', inplace=True)

    date_range = stock_data[max_stock].index

    # reindex for having the same size and fill null value with mean of prices
    for s, d in stock_data.items():
        if s != max_stock:
            d = d.reindex(date_range)
            mean_price = d['close'].mean()
            stock_data[s] = d.fillna(mean_price)


def plot_graph(stock_relationship, leaders, followers):
    threshold = 0.05

    # create a directed graph and add nodes and edges
    G = nx.DiGraph()
    G.add_nodes_from(stocks)

    node_colors = []
    for node in G.nodes():
        if node in leaders:
            node_colors.append('green')  # color for leaders
        elif node in followers:
            node_colors.append('red')  # color for followers
        else:
            node_colors.append('gray')  # color for others

    for i in stocks:
        for j in stocks:
            if i != j and abs(stock_relationship.at[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(stock_relationship.at[i, j]))

    persian_font = fm.FontProperties(fname='./Vazirmatn-Regular.ttf')

    # Draw the graph
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=30)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    for (u, v, w) in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w['weight'] * 10,
                               arrowstyle='->', arrowsize=50, edge_color='black')

    for node, (x, y) in pos.items():
        reshaped_text = arabic_reshaper.reshape(node)
        bidi_text = get_display(reshaped_text)
        plt.text(x, y, bidi_text, fontproperties=persian_font, fontsize=12, ha='center', va='center')

    plt.title("Stock Relationships Graph", fontsize=15)
    plt.axis('off')
    plt.show()


stocks = ['وغدیر', 'فباهنر', 'فسرب', 'شخارک', 'مداران', 'خکاوه', 'کمنگنز', 'خاور', 'وبهمن', 'بموتو', 'دکوثر', 'خرینگ',
          'وبشهر', 'ستران', 'ثامان', 'خمحرکه', 'وپخش', 'وصندوق', 'وپترو', 'وکار', 'فرآور', 'سکرما', 'دالبر', 'خساپا',
          'ومعادن', 'بکاب', 'بترانس', 'شنفت', 'سفارس', 'غپاک', 'شکربن', 'غبشهر', 'فلوله', 'شبهرن', 'سرود', 'کروی',
          'ونفت', 'کچاد', 'سپاها', 'دسبحا', 'وسپه', 'دلقما', 'وساخت', 'ثمسکن', 'وتوکا', 'دجابر', 'وصنعت', 'دکیمی',
          'ونوین', 'خصدرا', 'سشمال', 'سشرق', 'خاذین', 'شاراک', 'پسهند', 'شپترو', 'داسوه', 'شپاکسا', 'والبر', 'وصنا',
          'وبوعلی', 'وتوشه', 'خوساز', 'کهمدا', 'دپارس', 'خشرق', 'خمهر', 'وتوصا', 'وبانک', 'رانفور', 'ختوقا', 'لسرما',
          'ورنا', 'بنیرو', 'ونیکی', 'غشهد', 'درازک', 'خودرو', 'وبیمه', 'شگل', 'تایرا', 'لابسا', 'خزامیا']

SHOW_GRAPH = True

if __name__ == '__main__':
    data = read_from_csv(stocks)
    if not data:
        counter = 0
        # download the data
        while counter < 5:
            counter += 1

            try:
                data = tse.download(symbols=stocks, write_to_csv=True)
                if data:
                    break
            except Exception as e:
                print(e)
                continue
        else:
            quit()

    stocks = data.keys()

    prepare_data(data)

    # calculate the log return
    log_returns = {s: np.log(d['close']).diff().dropna() for s, d in data.items()}

    # finding the relationships based on pearson
    relationships = pd.DataFrame(index=stocks, columns=stocks)
    for i in data.keys():
        for j in data.keys():
            if i != j:
                correlation, _ = pearsonr(log_returns[i], log_returns[j])
                relationships.at[i, j] = correlation

    scores = relationships.sum(axis=1)

    sorted_scores = scores.sort_values(ascending=False)

    # set a threshold to define leaders and followers (you can adjust this)
    leader_threshold = np.percentile(scores, 80)  # Top 20% are leaders
    follower_threshold = np.percentile(scores, 20)  # Top 20% are followers

    leaders = set(scores[scores > leader_threshold].index)
    followers = set(scores[scores < follower_threshold].index)

    print(f"{'Category':<15}{'Score':<10}{'Stock'}")
    print('-' * 35)

    for stock, score in sorted_scores.items():
        category = '  Leader' if stock in leaders else 'Follower' if stock in followers else '   Other'
        print(f"{stock:<15}{score:<10.2f}{category}")

    if SHOW_GRAPH:
        plot_graph(relationships, leaders, followers)

    results = trading_model(data, list(leaders), list(followers), 100000000)

    if SHOW_GRAPH:
        plot_profit(results['months'], results['holdings'])