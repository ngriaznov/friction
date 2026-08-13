"""Same shape as the Java demo, so the two frontends can be compared."""

import logging

LOG = logging.getLogger("orders")

TOTALS = {}


def process(prices, tax_rate, apply_tax, order_id):
    LOG.info("processing %d prices", len(prices))

    unused = "this value goes nowhere"
    inspected = 0

    subtotal = 0.0
    for price in prices:
        if price is None or price < 0:
            continue
        subtotal += price
        inspected += 1

    total = subtotal
    if apply_tax:
        total = subtotal * (1.0 + tax_rate)

    LOG.debug("inspected %d entries", inspected)

    TOTALS[order_id] = total
    return total
