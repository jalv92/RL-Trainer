[![Apex Trader Funding Help Center home page](https://support.apextraderfunding.com/hc/theming_assets/01JB15B1TZ64VG5EA4WZWC8WCE)](https://apextraderfunding.com/)Toggle navigation menu

[Contact Us](https://support.apextraderfunding.com/hc/en-us/requests/new) [FAQ](https://support.apextraderfunding.com/hc/en-us) [Login to Apex Trader Funding](https://apextraderfunding.com/member/member/) [Sign in](https://support.apextraderfunding.com/hc/en-us/signin?return_to=https%3A%2F%2Fsupport.apextraderfunding.com%2Fhc%2Fen-us%2Farticles%2F31519769997083-Evaluation-Rules "Opens a dialog")

![](https://www.google.com/images/cleardot.gif)[Select Language![](https://www.google.com/images/cleardot.gif)​![](https://www.google.com/images/cleardot.gif)▼](https://support.apextraderfunding.com/hc/en-us/articles/31519769997083-Evaluation-Rules#)

# Help Center

## Search

1. [Apex Trader Funding](https://support.apextraderfunding.com/hc/en-us)
2. [Evaluation Accounts (EA)](https://support.apextraderfunding.com/hc/en-us/sections/31319671978395-Evaluation-Accounts-EA)

# Evaluation Rules

## Trailing Drawdown and Rules

### **Understanding the Trailing Threshold and Evaluation Rules (Master Course)**

**Introduction**

This guide covers how the trailing threshold works with trading evaluation and funded accounts, including specific details for RITHMIC and TRADOVATE accounts. We’ll explain trailing drawdowns, max drawdowns, and evaluation rules and provide tips for monitoring your account effectively.

The video below summarises the Trailing Drawdown Threshold:

Trailing Threshold Drawdown - Apex Trader Funding - YouTube

[Photo image of Apex Trader Funding](https://www.youtube.com/channel/UC0F1ZMdysGBgGCMpdxVyGGQ?embeds_widget_referrer=https%3A%2F%2Fsupport.apextraderfunding.com%2Fhc%2Fen-us%2Farticles%2F31519769997083-Evaluation-Rules&embeds_referring_euri=https%3A%2F%2Fsupport.apextraderfunding.com%2F&embeds_referring_origin=https%3A%2F%2Fsupport.apextraderfunding.com)

Apex Trader Funding

26.1K subscribers

[Trailing Threshold Drawdown - Apex Trader Funding](https://www.youtube.com/watch?v=9u0n4J-5Q6c)

Apex Trader Funding

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

### **Key Concepts**

- **Starting Balance**: Your account starts with an initial balance, and each plan has a specified maximum number of contracts you can trade.
- **No Daily Max Drawdown**: There is no daily maximum drawdown limit.
- **Max Loss Drawdown**: Each plan has a varying max loss drawdown. For example, a $50k plan starts at $47,500, which is $2,500 below the initial $50k balance.

### **Trailing Drawdown Example**

To better understand how the trailing threshold works, let’s go through a practical example:

- **Starting Balance**: $50,000
- **Allowed Drawdown**: $2,500, meaning your threshold begins at $47,500.

  - **First Trade**: You make $500 and close the trade. Now, your balance is $50,500, and your threshold will update to $48,000 (the remaining $2,500 below your new balance).
  - **Second Trade**: During the next trade, your balance peaks at $50,875, but you only close the trade at $50,100. Your threshold will be $48,375, trailing $2,500 behind the highest balance reached ($50,875) even though you closed at a lower amount.

![](https://support.apextraderfunding.com/hc/article_attachments/31829077782299)

This example highlights that the trailing threshold is based on the highest live value during trades, not on closed trade values. It’s crucial to remember this to avoid unintentionally exceeding the maximum drawdown.

## **Drawdown Types: FULL vs. STATIC Accounts**

- **FULL Accounts:**

  - In paid/funded accounts, the max drawdown stops trailing once the liquidation threshold is the initial balance plus $100 (e.g., $50,100 on a 50k plan).
  - The trailing threshold stops moving once your unrealized balance exceeds the plan’s start balance by the drawdown amount plus $100.
  - The example below illustrates a trailing drawdown level at 25 points using 5 contracts on NQ, which would be around the $2500 trailing drawdown for the $50,000 account.

![FINAL TDD.gif](https://support.apextraderfunding.com/hc/article_attachments/31829816191131)

- **STATIC Accounts:**

  - The drawdown level remains unchanged. For instance, a 100k static account has a set drawdown at $99,375.
  - The example below illustrates a fixed drawdown level at roughly 10.5 points using 3 contracts on NQ, which would be around the $625 static drawdown for the $100,000 account.

![FINAL STATIC.gif](https://support.apextraderfunding.com/hc/article_attachments/31829844325147)

### **Trailing Max Drawdown by Plan and Contract Size**

**Plan** **Account Size** **Max Loss/Trailing Threshold**

25K Full Size 4 Minis - 1500

50K Full Size 10 Minis - 2500

75K Full Size 12 Minis - 2750

100K Full Size 14 Minis - 3000

150K Full Size 17 Minis - 5000

250K Full Size 27 Minis - 6500

300K Full Size 35 Minis - 7500

100K Static 2 Minis - 625

### **Special Rules for RITHMIC and TRADOVATE Accounts For Trailing Drawdown**

- **RITHMIC Accounts**: During the evaluation, the drawdown stops trailing when the threshold balance reaches your profit target (not the account balance). For instance, a 50K account stops trailing when the threshold reaches $53,000.
- **TRADOVATE Accounts**: The trailing drawdown continues to trail in evaluations and does not stop.

### **Monitoring Your Account and Avoiding Liquidation**

- **Max Drawdown Monitoring**: Always check your trailing max drawdown in your RTrader or Tradovate dashboard. The auto-liquidation threshold reflects this value.
- **Drawdown Failure**: Dropping below the drawdown threshold fails the evaluation.
- **Rithmic and Tradovate Access**: For RITHMIC accounts, download RTrader Pro from the member area. Both RTrader and Tradovate platforms allow you to monitor and control your account.

## **Evaluation Account Rules**

1. **Account Balance**: Your balance must close at or above the profit goal without hitting the max drawdown. Ensure a clear understanding of the Trailing Threshold Drawdown to prevent accidental liquidation.
2. **Minimum Trading Days**: Complete a minimum of seven trading days (non-consecutive) for a valid evaluation (subject to any active promotions for 1-day-pass).
3. **Professional Conduct**: Follow the code of conduct and maintain professionalism. Sharing your username or password is prohibited.
4. **Account Monitoring**: Keep RTrader/Tradovate open to monitor your balance and max drawdown, and as a backup for closing trades if needed. Stay aware of the trailing threshold.
5. **Trailing Threshold Awareness**: Track the trailing threshold carefully, as it moves with the highest profit point achieved in active trades and can impact your evaluation outcome if not managed

### **Evaluation Account General Guidelines**

- ### **Trade Close-Out Timing**:

  - **All trades must be closed, and all pending orders canceled by 4:59 PM ET**. Holding trades through the close is not permitted.
  - APEX has a safeguard to automatically close open positions and cancel pending orders attached to a position at 4:59 PM ET. This safeguard is a final resort and should not be relied upon. Therefore, you must manually cancel orders not attached to a position. If **they remain open, they may liquidate your account**.
  - Markets requiring an earlier close are unaffected by this rule and must still be closed manually by the trader.
- ### **Holiday Trading**:

  - **Holidays and Half-Day Closures**: You can trade on holidays if the market is open; however, half-day holidays do not count as trading days. This trading day is combined with the next trading day.
  - **Important Note**: During early holiday closures, trades must be closed at the market’s designated earlier time.
- ### **Automatic Close-Out Safeguard**:

  - The 4:59 PM rule is a fail-safe, not a primary trade-close tool. **It’s your responsibility to close all trades before 4:59 PM ET**.
  - Leaving trades open may result in gaps that affect your threshold, potentially causing account failure. Relying solely on the auto-close feature is risky.
- ### **Maintaining the Profit Goal**:

  - If you reach the profit goal before completing the minimum trading days, ensure your balance stays above the goal until the minimum trading requirement is met.
- ### **Account Status & Renewal**:

  - **Failed Account Reset**: Accounts that fail and aren’t reset within eight days will be disabled if there are no active subscriptions. Reset the account or open a new one to continue.
  - **Negative Balance on Renewal**: A negative initial balance upon renewal doesn’t mean a failed evaluation; only a negative drawdown will constitute a failure. Accounts in a Failed or Blown state will receive a reset upon renewal.

**Reminder**: Please review the specific rules of Apex Trader Funding rather than assuming similarities with other companies. These evaluation guidelines are designed to help you meet the standards Apex sets for successful account funding.

### **Consistency and Planning for Success**

- **Trading Consistency**: Randomly changing trade sizes or going “all in” undermines long-term success. Following a consistent trading plan helps you grow your account steadily.
- **Consistency Rules**: These apply mainly to PA and Funded accounts, where steady, structured trading is required.

**Important:** The trailing threshold provides a safety margin, but understanding its dynamics and monitoring it in real time is crucial to avoid unintended account liquidation.

For a detailed walkthrough, watch our video explaining the Trailing Threshold below.

Trailing Threshold Drawdown - Apex Trader Funding - YouTube

[Photo image of Apex Trader Funding](https://www.youtube.com/channel/UC0F1ZMdysGBgGCMpdxVyGGQ?embeds_widget_referrer=https%3A%2F%2Fsupport.apextraderfunding.com%2Fhc%2Fen-us%2Farticles%2F31519769997083-Evaluation-Rules&embeds_referring_euri=https%3A%2F%2Fsupport.apextraderfunding.com%2F&embeds_referring_origin=https%3A%2F%2Fsupport.apextraderfunding.com)

Apex Trader Funding

26.1K subscribers

[Trailing Threshold Drawdown - Apex Trader Funding](https://www.youtube.com/watch?v=9u0n4J-5Q6c)

Apex Trader Funding

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on www.youtube.com](https://www.youtube.com/watch?v=9u0n4J-5Q6c)

Watch on

## **Common Helpdesk Questions**

These comprehensive answers are related to frequently asked questions, which will help you navigate our platform effectively. Please read carefully to fully understand our policies and procedures.

### **Platform Conversion**

**Can I convert from Rithmic to Tradovate or vice versa?**

**No**, the account setups for Rithmic and Tradovate are different, and each platform has its own fees. If you wish to switch platforms, you must sign up for a new plan specific to that platform.

**How important is it that I sign up for the right plan?**

It is **paramount** that you choose the correct plan on our website or dashboard. If you select the wrong plan or platform, there will be **no refunds**, with no exceptions. We incur charges for every new account setup and cannot issue refunds for user mistakes. **Please read all options carefully before you click to pay for your account!**

### **Evaluation Trading Accounts**

**Can I take the evaluation trading in SIM, or do I have to trade with live money?**

All evaluations are conducted using **simulation (SIM) accounts**. You will log in with your Rithmic and/or Tradovate username and password, then select your evaluation account(s) for trading.

**During the evaluation period, can I trade my evaluations, PAs, and my own account simultaneously?**

Yes, if you have your own NinjaTrader license key, you can trade your evaluations, Performance Accounts (PAs), and your personal account at the same time.

### **Trading Frequency During Evaluation**

**Do I have to trade every day during my evaluation time?**

**No**, you are not required to trade every day. You can take days off as needed. Trades can be spread out over time, as long as you have a minimum of **seven traded days** to qualify. There are no set time restrictions for eligibility.

- **Recurring Payments**: Payments are made every 30 days for as long as you need the evaluation account.
- **Resets**: You can reset the account as many times as necessary. Remember, you must have traded a minimum of seven days to qualify, regardless of how long it takes (unless there is an active promotion, including 1-day pass).

## **Maximum Position Size on Regular Contracts and Micros**

**What is the max position size on regular contracts and micros?**

**Max Position Size**: The maximum position size is limited by the number of contracts specified in your chosen plan. This applies to all instruments and positions. For example, if your maximum is 10 contracts, you could trade 7 contracts on ES and 3 contracts on GC simultaneously, totaling 10 contracts. Orders exceeding this limit will be rejected.

**Micros**: You can trade micro contracts up to the maximum contract size listed for your plan. Available micro futures include:

**Micro Futures** **Symbol** **Exchange**

Micro E-Mini S&P 500 MES CME

Micro E-Mini Dow Jones MYM CME

Micro E-Mini Nasdaq-100 MNQ CME

Micro E-Mini Russell 2000 M2K CME

E-Micro Gold MGC CME

E-Micro AUD/USD M6A CME

E-Micro EUR/USD M6E CME

E-Micro USD/JPY M6J CME

Micro Crude Oil MCL NYMEX

## **Starting Your Evaluation**

### **How many profitable weeks should I have before I’m ready for an evaluation?**

You can **start now**! There is no requirement to have a certain number of profitable weeks before beginning an evaluation. In fact, our program offers an economical way to learn to trade and risk management without risking your own capital.

- **Learning Opportunity**: Use the evaluation to develop your trading skills and strategies.
- **Risk Management**: Avoid risking personal funds needed for essential expenses like retirement or education.
- **Leverage**: For a small monthly fee, you can leverage substantial capital with the potential to receive payouts.

### **Age Requirement**

**What is the age requirement for having an evaluation and a funded account?**

You must be **18 years old or older** to participate in our evaluations and funded accounts. Breaching this rule will result in a permanent ban on using Apex services in the future.

### **Trading Days and Holidays**

**Do Sundays and holidays count as trading days?**

• **Sundays**: Trading on Sundays counts as part of **Monday’s trading day**. A trading day is defined as 6:00 PM ET one day until 4:59 PM ET the next day.

• **Holidays**: You can trade on holidays if the market is open. However, **half-day holidays do not count** as a trading day.

### **Account Types: Static vs. Full**

**What is the difference between Static and Full Accounts?**

• **Static Accounts**: These accounts have a fixed trailing threshold that does not adjust with your account balance.

• **Full Accounts**: These accounts have a trailing threshold that adjusts with your account balance and allows for larger contract sizes and stop-loss limits.

### **Account Activation Timing**

**Can I sign up today and start 10 days later?**

Yes, you can sign up today and begin trading at a later date. However, please note that your **monthly billing cycle starts on the day you sign up**.

### **Trading Style**

**Do I continue trading the way I am now, or do I have to trade another way?**

You can continue trading in your current style. However, to comply with our end-of-day policies, you must ensure that all your trades are closed by 4:59 PM ET each day.

### **Evaluation Time Frame**

**Do I have to complete the evaluation in a certain number of days?**

You need to log **seven trading days** as part of the criteria to pass the evaluation, but there is **no maximum time limit**. You can take as long as you need to qualify.

### **NinjaTrader License Key**

**Do I have to use my own NinjaTrader license key?**

**No**, you don’t have to use your own key. We provide you with a NinjaTrader license key if needed.

**Profit Target Trading Days (Outside of 1-day-pass promotions)**

**Seven trading days are the minimum required days for the profit target. What is the maximum?**

There is **no maximum**. You can take as long as you need to qualify. To pass the evaluation, you must:

- Trade at least **seven separate trading days**
- Meet the **profit goal**
- Follow all **trading rules**
- Demonstrate **consistent trading ability**

### **Professional Status and Data Fees**

**What is the difference between “non-professional” and “professional”?**

- **Non-Professional**: Most traders are non-professional, and your data fees are included in your plan.

- **Professional**: Selecting ‘Professional’ incurs extra fees. For CME professional data, fees are **$115 or more per calendar month per exchange**, in addition to other fees. Only select ‘Professional’ when setting up Rithmic if you are truly a professional trader.

![](https://support.apextraderfunding.com/hc/article_attachments/31829050202523)

**Important**: Choosing ‘Professional’ unnecessarily will result in significant additional costs.

### **Margin Calls**

**Who gets the margin call if it happens?**

**You do not receive margin calls**. Apex Trader Funding handles all margin requirements internally.

### **Profit Goals and Commissions**

**Is the profit goal net of commissions after all in costs (profits and losses)?**

**Yes**, the profit goal is net of commissions and all associated costs. You can view your real-time profit and loss (PnL) and account balances in RTrader or the Tradovate web app.

### **Account Violations and Resets**

**What happens if I violate a trading rule?**

If you receive an error message indicating a rule violation (e.g., surpassing the maximum drawdown), your account may be disabled.

- **Evaluation Accounts**: You will need to **reset your account** or start a new evaluation. Resets allow you to start over with the full balance and trailing threshold.
- **Performance Accounts (PAs)**: **Resets are not available**. Violations in a PA account may lead to account closure.

## **Trading Multiple Accounts**

**Why would I want to trade multiple evaluation accounts?**

Trading multiple accounts can diversify your strategies and increase potential profits. For a detailed explanation, please refer to our video here:

Multiple Accounts-22 - YouTube

[Photo image of Apex Trader Funding](https://www.youtube.com/channel/UC0F1ZMdysGBgGCMpdxVyGGQ?embeds_widget_referrer=https%3A%2F%2Fsupport.apextraderfunding.com%2Fhc%2Fen-us%2Farticles%2F31519769997083-Evaluation-Rules&embeds_referring_euri=https%3A%2F%2Fsupport.apextraderfunding.com%2F&embeds_referring_origin=https%3A%2F%2Fsupport.apextraderfunding.com)

Apex Trader Funding

26.1K subscribers

[Multiple Accounts-22](https://www.youtube.com/watch?v=kDYJV-nJG5A)

Apex Trader Funding

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on www.youtube.com](https://www.youtube.com/watch?v=kDYJV-nJG5A)

Watch on

### **Data Fees Responsibility**

**Am I responsible for data fees?**

- **Level 1 Data**: Level 1 data (L1) is included in your fee; there are no additional charges.
- **Depth of Market (DOM)**:

  - **Rithmic**: If you purchase DOM through Rithmic, it will **expire on the last day of the month** and will not auto-renew. You must manually purchase it again if needed, regardless of the purchase date.
  - **Tradovate**: If you purchase additional data feeds through Tradovate, you will need to **cancel those feeds directly through Tradovate** if you no longer wish to be billed.

## Eval Charts Tutorial and Troubleshooting

Our Evaluation and PA Charts provide a clear visual overview of your trading performance and account status. On the left-hand side, the ‘Eval Charts’ or ‘PA Charts’ section in the sidebar is where you access these detailed graphs and tables.

At the top, you’ll see a cumulative Profit and Loss (PnL) chart, which shows how your account’s profits or losses have changed over time, measured in trading days.

Directly underneath, each row of the table displays key account metrics updated nightly at midnight ET. These include your most recent trade date, account type, total PnL, and the number of trading days completed. You can also see critical thresholds such as the trailing stop (maximum allowed drawdown), current account balance, target profit levels, and an easy-to-read progress bar that visually indicates how close you are to reaching the goals or potential breaches.

Finally, the status and state columns inform you if the account is active, inactive, passed, or blown, ensuring you always know exactly where you stand in your evaluation or PA journey.

![](https://support.apextraderfunding.com/hc/article_attachments/31829050202779)

### **Need Assistance?**

If you have any additional questions or need further clarification, please don’t hesitate to submit a help desk ticket. We’re here to help you succeed.

**To submit a help desk ticket, [CLICK HERE](https://support.apextraderfunding.com/hc/en-us/requests/new)**

### Can't find what you're looking for?


Our team of experts is here to help



[Contact us](https://support.apextraderfunding.com/hc/en-us/requests/new)

[Back to website](https://apextraderfunding.com/index.html)

[💬 Need Help?](https://support.apextraderfunding.com/hc/requests/new)

![](https://fonts.gstatic.com/s/i/productlogos/translate/v14/24px.svg)

Original text

Rate this translation

Your feedback will be used to help improve Google Translate