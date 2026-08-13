import java.util.List;
import java.util.logging.Logger;

/** A deliberately ordinary method mixing real behavior with noise. */
public class OrderProcessor {
    private static final Logger LOG = Logger.getLogger("orders");

    private double runningTotal;

    public double process(List<Double> prices, double taxRate, boolean applyTax) {
        LOG.info("processing " + prices.size() + " prices");

        String unused = "this value goes nowhere";
        int inspected = 0;

        double subtotal = 0.0;
        for (Double price : prices) {
            if (price == null || price < 0) {
                continue;
            }
            subtotal += price;
            inspected++;
        }

        double total = subtotal;
        if (applyTax) {
            total = subtotal * (1.0 + taxRate);
        }

        LOG.fine("inspected " + inspected + " entries");

        this.runningTotal = total;
        return total;
    }
}
