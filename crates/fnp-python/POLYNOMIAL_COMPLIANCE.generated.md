# fnp_python polynomial Compliance Matrix (auto-generated)

> Source: `scripts/fnp_polynomial_compliance_matrix.py`  
> Target: `numpy.polynomial.*` public API (numpy 2.0+ snapshot)  
> Do not edit by hand — regenerate via the script.


## Summary by RequirementLevel

| Level | Present | Total | Coverage |
|-------|--------:|------:|---------:|
| MUST | 66 | 66 | 100.0% |
| SHOULD | 34 | 34 | 100.0% |
| MAY | 0 | 0 | 0.0% |
| **Total** | **100** | **100** | **100.0%** |

## Per-subtree coverage

| Subtree | Present | Total | Coverage |
|---------|--------:|------:|---------:|
| `chebyshev` | 16 | 16 | 100.0% |
| `class` | 6 | 6 | 100.0% |
| `hermite` | 16 | 16 | 100.0% |
| `hermite_e` | 16 | 16 | 100.0% |
| `laguerre` | 16 | 16 | 100.0% |
| `legendre` | 16 | 16 | 100.0% |
| `polynomial` | 14 | 14 | 100.0% |

## Present names

| Name | Level | Subtree |
|------|:-----:|---------|
| `Chebyshev` | MUST | `class` |
| `Hermite` | MUST | `class` |
| `HermiteE` | MUST | `class` |
| `Laguerre` | MUST | `class` |
| `Legendre` | MUST | `class` |
| `Polynomial` | MUST | `class` |
| `cheb2poly` | SHOULD | `chebyshev` |
| `chebadd` | MUST | `chebyshev` |
| `chebder` | MUST | `chebyshev` |
| `chebdiv` | MUST | `chebyshev` |
| `chebfromroots` | MUST | `chebyshev` |
| `chebint` | MUST | `chebyshev` |
| `chebline` | SHOULD | `chebyshev` |
| `chebmul` | MUST | `chebyshev` |
| `chebmulx` | SHOULD | `chebyshev` |
| `chebpow` | MUST | `chebyshev` |
| `chebroots` | MUST | `chebyshev` |
| `chebsub` | MUST | `chebyshev` |
| `chebtrim` | SHOULD | `chebyshev` |
| `chebval` | MUST | `chebyshev` |
| `chebvander` | SHOULD | `chebyshev` |
| `herm2poly` | SHOULD | `hermite` |
| `hermadd` | MUST | `hermite` |
| `hermder` | MUST | `hermite` |
| `hermdiv` | MUST | `hermite` |
| `herme2poly` | SHOULD | `hermite_e` |
| `hermeadd` | MUST | `hermite_e` |
| `hermeder` | MUST | `hermite_e` |
| `hermediv` | MUST | `hermite_e` |
| `hermefromroots` | MUST | `hermite_e` |
| `hermeint` | MUST | `hermite_e` |
| `hermeline` | SHOULD | `hermite_e` |
| `hermemul` | MUST | `hermite_e` |
| `hermemulx` | SHOULD | `hermite_e` |
| `hermepow` | MUST | `hermite_e` |
| `hermeroots` | MUST | `hermite_e` |
| `hermesub` | MUST | `hermite_e` |
| `hermetrim` | SHOULD | `hermite_e` |
| `hermeval` | MUST | `hermite_e` |
| `hermevander` | SHOULD | `hermite_e` |
| `hermfromroots` | MUST | `hermite` |
| `hermint` | MUST | `hermite` |
| `hermline` | SHOULD | `hermite` |
| `hermmul` | MUST | `hermite` |
| `hermmulx` | SHOULD | `hermite` |
| `hermpow` | MUST | `hermite` |
| `hermroots` | MUST | `hermite` |
| `hermsub` | MUST | `hermite` |
| `hermtrim` | SHOULD | `hermite` |
| `hermval` | MUST | `hermite` |
| `hermvander` | SHOULD | `hermite` |
| `lag2poly` | SHOULD | `laguerre` |
| `lagadd` | MUST | `laguerre` |
| `lagder` | MUST | `laguerre` |
| `lagdiv` | MUST | `laguerre` |
| `lagfromroots` | MUST | `laguerre` |
| `lagint` | MUST | `laguerre` |
| `lagline` | SHOULD | `laguerre` |
| `lagmul` | MUST | `laguerre` |
| `lagmulx` | SHOULD | `laguerre` |
| `lagpow` | MUST | `laguerre` |
| `lagroots` | MUST | `laguerre` |
| `lagsub` | MUST | `laguerre` |
| `lagtrim` | SHOULD | `laguerre` |
| `lagval` | MUST | `laguerre` |
| `lagvander` | SHOULD | `laguerre` |
| `leg2poly` | SHOULD | `legendre` |
| `legadd` | MUST | `legendre` |
| `legder` | MUST | `legendre` |
| `legdiv` | MUST | `legendre` |
| `legfromroots` | MUST | `legendre` |
| `legint` | MUST | `legendre` |
| `legline` | SHOULD | `legendre` |
| `legmul` | MUST | `legendre` |
| `legmulx` | SHOULD | `legendre` |
| `legpow` | MUST | `legendre` |
| `legroots` | MUST | `legendre` |
| `legsub` | MUST | `legendre` |
| `legtrim` | SHOULD | `legendre` |
| `legval` | MUST | `legendre` |
| `legvander` | SHOULD | `legendre` |
| `poly2cheb` | SHOULD | `chebyshev` |
| `poly2herm` | SHOULD | `hermite` |
| `poly2herme` | SHOULD | `hermite_e` |
| `poly2lag` | SHOULD | `laguerre` |
| `poly2leg` | SHOULD | `legendre` |
| `polyadd` | MUST | `polynomial` |
| `polyder` | MUST | `polynomial` |
| `polydiv` | MUST | `polynomial` |
| `polyfromroots` | MUST | `polynomial` |
| `polyint` | MUST | `polynomial` |
| `polyline` | SHOULD | `polynomial` |
| `polymul` | MUST | `polynomial` |
| `polypow` | MUST | `polynomial` |
| `polyroots` | MUST | `polynomial` |
| `polysub` | MUST | `polynomial` |
| `polytrim` | SHOULD | `polynomial` |
| `polyval` | MUST | `polynomial` |
| `polyvalfromroots` | SHOULD | `polynomial` |
| `polyvander` | SHOULD | `polynomial` |
