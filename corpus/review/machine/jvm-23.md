Nice set of tests. Using `@ParameterizedTest` with `@CsvSource` for `DiscountCalculatorTest.appliesPercentageDiscountCorrectly` is a good fit here — six rows covering different percentage tiers and rounding edge cases is much easier to read and extend than six near-identical `@Test` methods would have been, and the boundary cases (0%, 100%, and the odd-cents rounding case at 33%) are exactly the ones I'd have asked for if they were missing.

Two small things:

The `@CsvSource` rows don't have display names, so a failure in CI shows up as `appliesPercentageDiscountCorrectly[3]` rather than telling you which input actually failed without going and counting rows. Adding a `name` attribute to `@ParameterizedTest` (e.g., `"{0}% off {1} = {2}"`) makes failures self-explanatory at a glance, which is worth the one-line change given how much time it saves when this test fails in someone else's CI run six months from now.

`MockitoExtension` is used correctly for `PricingPolicyProvider`, and the `when(policyProvider.getMaxDiscountPercent()).thenReturn(50)` stub is scoped appropriately to just the test that needs the cap-enforcement behavior rather than applied globally — good instinct not to over-stub in `@BeforeEach` for a value only one test actually needs.

Approving as-is — these are solid, readable tests and the parameterization choice is the right one for this kind of tiered-calculation logic. The display-name suggestion is a nit, not a blocker.
