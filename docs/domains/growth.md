# Growth

::: hcsim.engine.domains.growth

## Routing and sizing

::: hcsim.engine.supply

## Data findings

- The growth arms can't be told apart: WAPE scores only the mean, which the Poisson and
  negative-binomial arms share. A count likelihood would separate them.
- With the fill yield at its default of 1, the `expected_fill` rule sizes exactly as `strength`
  does, and supply alignment counts every open opening as a head to come. On the tiny world, it
  undershoots every target and trails no alignment (issue #634).
