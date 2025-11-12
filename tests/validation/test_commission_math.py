"""Test to verify commission mathematics"""

# Simulate the first trade
initial_cash = 100000.00
entry_price = 28894.68
commission_rate = 0.001  # 0.1%

# Calculate order quantity
order_qty = initial_cash / entry_price
print(f"Order quantity: {order_qty:.10f} BTC")

# Calculate max affordable with commission
# We need: qty * price * (1 + commission_rate) <= cash
max_affordable = initial_cash / (entry_price * (1 + commission_rate))
print(f"Max affordable: {max_affordable:.10f} BTC")

# Calculate costs
notional = max_affordable * entry_price
commission = notional * commission_rate
total_cost = notional + commission
remaining_cash = initial_cash - total_cost

print(f"\nEntry fill:")
print(f"  Quantity: {max_affordable:.10f} BTC")
print(f"  Notional: ${notional:.10f}")
print(f"  Commission: ${commission:.10f}")
print(f"  Total cost: ${total_cost:.10f}")
print(f"  Remaining cash: ${remaining_cash:.10f}")

# Now simulate the exit
exit_price = 28804.54
exit_qty = max_affordable  # Try to sell entire position

exit_notional = exit_qty * exit_price
exit_commission = exit_notional * commission_rate
exit_proceeds = exit_notional - exit_commission
final_cash = remaining_cash + exit_proceeds

print(f"\nExit fill (attempting to sell {exit_qty:.10f} BTC):")
print(f"  Quantity: {exit_qty:.10f} BTC")
print(f"  Notional: ${exit_notional:.10f}")
print(f"  Commission: ${exit_commission:.10f}")
print(f"  Net proceeds: ${exit_proceeds:.10f}")
print(f"  Final cash: ${final_cash:.10f}")

# Check if this matches observed behavior
print(f"\nObserved in test:")
print(f"  Cash after entry: $97.66")
print(f"  Cash after exit: $99262.25")
print(f"  Position after entry: 3.4537 BTC")
print(f"  Position after exit: 0.0072 BTC")

# Calculate what was actually sold
cash_increase = 99262.25 - 97.66
# If X BTC sold: X * price * (1 - commission_rate) = cash_increase
actual_sold = cash_increase / (exit_price * (1 - commission_rate))
leftover_theoretical = max_affordable - actual_sold

print(f"\nCalculated from observed cash (using theoretical fill):")
print(f"  Actually sold: {actual_sold:.10f} BTC")
print(f"  Leftover: {leftover_theoretical:.10f} BTC")

# Now use the OBSERVED fill quantity
observed_fill = 3.4537
leftover_observed = observed_fill - actual_sold

print(f"\nUsing observed entry fill (3.4537 BTC):")
print(f"  Entry fill: {observed_fill:.10f} BTC")
print(f"  Actually sold: {actual_sold:.10f} BTC")
print(f"  Leftover: {leftover_observed:.10f} BTC")
print(f"  Leftover matches observed 0.0072: {abs(leftover_observed - 0.0072) < 0.0001}")

# Verify entry math with observed fill
observed_notional = observed_fill * entry_price
observed_commission = observed_notional * commission_rate
observed_total = observed_notional + observed_commission
observed_remaining = initial_cash - observed_total

print(f"\nEntry with observed fill 3.4537 BTC:")
print(f"  Notional: ${observed_notional:.2f}")
print(f"  Commission: ${observed_commission:.2f}")
print(f"  Total cost: ${observed_total:.2f}")
print(f"  Remaining cash: ${observed_remaining:.2f}")
print(f"  Matches observed $97.66: {abs(observed_remaining - 97.66) < 1.0}")
