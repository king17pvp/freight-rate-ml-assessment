# DOMAIN_KNOWLEDGE.md — US Truckload Freight & Spot Rates

Companion to [KNOWLEDGE.md](KNOWLEDGE.md) (modeling theory). This file covers the *domain*:
what a freight rate is, who sets it, what moves it, and how each concept maps to the columns
in `train-test.csv`. Where our (synthetic) dataset deviates from real-world behavior, that's
called out explicitly — deviations are worth mentioning in the report.

---

## 1. The problem we're actually modeling

We're predicting the **posted rate** for full-truckload (FTL) shipments on the **spot
market**: the price offered to move one entire trailer-load from city A to city B on a given
date. This is Spotter's world — brokers/carriers deciding in real time what a load should
pay. A model that predicts this well is a pricing engine: it tells a broker what to quote and
a carrier whether an offered load is above or below market.

**Spot vs. contract:** contract rates are negotiated annually between shippers and carriers;
spot rates are one-off prices for individual loads, set by immediate supply and demand.
Spot is far more volatile — it's the market's live temperature. Our data (one price per load
per day) is spot-shaped.

**Posted rate specifically:** the price attached to a load listing on a load board (DAT,
Truckstop.com). It's an *offer*, not necessarily the settled price — but for modeling
purposes it's the market signal we predict.

---

## 2. Anatomy of a rate — the formulas

**The all-in rate decomposes as:**

```
all-in rate = linehaul + fuel surcharge (FSC) + accessorials
rate per mile (RPM) = all-in rate / miles          ← the industry's universal unit
```

- **Linehaul** — the base transportation charge; the part that moves with market supply/demand.
- **Fuel surcharge** — compensates fuel price swings, typically pegged to the DOE weekly
  diesel index: `FSC/mile ≈ (diesel price − peg price) / truck MPG` (peg ≈ $1.20–1.25,
  MPG ≈ 6–6.5). In 2025 van FSC averaged ≈ $0.40/mile.
- **Accessorials** — extras: detention (waiting), layover, driver assist, liftgate, etc.

**Benchmark numbers to calibrate intuition (national averages):**

| Quantity | Value | Source period |
|---|---|---|
| Dry van spot all-in | ≈ $2.00–2.10/mi (2025), $2.93/mi (Aug 2026) | DAT |
| Reefer spot all-in | ≈ $3.38/mi (Aug 2026) | DAT |
| Flatbed spot all-in | ≈ $3.54/mi (Aug 2026) | DAT |
| Carrier operating cost | $2.26/mi (2024), $2.336/mi (2025) | ATRI |
| Non-fuel operating cost | $1.854/mi (2025) | ATRI |

The ATRI cost line matters conceptually: **operating cost is a soft floor under rates**.
When spot rates sit below cost (as in the 2023–25 "freight recession"), carriers exit,
capacity shrinks, and rates eventually recover — the *capacity cycle* that drives multi-year
rate waves (the 2021 COVID spike being the extreme case KNOWLEDGE.md §1.2 warns about).

**Our data:** median ≈ $2.14/mile with no separate FSC column — `posted_rate` is all-in.
That's consistent with 2025 dry-van spot levels, so the simulation is calibrated to reality.

---

## 3. Equipment types

The trailer type determines cost, driver skill, and the demand pool:

- **Dry van** — the standard enclosed 53' box; the commodity workhorse (majority of loads);
  cheapest.
- **Reefer** (refrigerated) — insulated trailer with a diesel cooling unit; hauls produce,
  meat, dairy, pharma. Higher equipment + fuel cost, temperature liability, and seasonal
  demand spikes → prices above van.
- **Flatbed** — open deck for machinery, steel, lumber; requires tarping/securement skill and
  is exposed to construction/industrial seasonality → prices above van, near reefer.

Real-world ordering is **Reefer ≈ Flatbed > Dry Van**, exactly what our EDA found
(Reefer > Flatbed > Dry Van in mean and median). Dry Van is 57% of our rows — also realistic.

---

## 4. Lane economics — the geometry concepts

- **Lane** — an origin→destination pair. THE unit of freight pricing; rates are quoted and
  benchmarked per lane. Direction matters: Atlanta→Chicago and Chicago→Atlanta are different
  lanes with different prices.
- **Headhaul / backhaul** — freight flows are imbalanced. Into a freight-poor region, the
  truck likely returns empty, so the inbound leg (headhaul) is priced to subsidize the return
  (backhaul, often priced near marginal cost). Classic example: Florida inbound is expensive,
  Florida outbound (outside produce season) is cheap.
- **Deadhead** — empty miles driven to reach the next pickup. Not directly paid, but priced
  into what carriers accept: a load delivering into a dead area must pay more.
- **Load-to-truck ratio** — posted loads ÷ posted trucks in a market (DAT publishes this);
  the standard local supply/demand gauge. High ratio → carriers' market → rates rise.
- **Tender rejection rate** — share of contracted loads carriers refuse (SONAR's OTRI index);
  when carriers reject contract freight, spot demand and spot rates jump.
- **Length-of-haul effect** — real-world RPM *falls* with distance: fixed costs (loading,
  unloading, dwell) amortize over more miles, so a 100-mile load might pay $4–5/mi while a
  2,000-mile load pays close to the national average. **Our data deviates here**: $/mile is
  nearly flat across distance (median $2.14, IQR $1.98–2.34) — the simulator prices
  near-linearly in miles. Good for modeling (distance enters ~linearly), but not a real-world
  pattern to claim in the report.
- **Miles are not haversine** — pricing uses routed road miles ("practical miles", or the
  older HHG short-line miles). Road ÷ great-circle ("circuity") ≈ 1.15–1.3. Our data's median
  circuity is 1.18 — realistic — with a floor at exactly 70 miles (minimum-charge behavior:
  real tariffs also floor very short moves).

**Our EDA echo:** forward/reverse lane $/mile correlate only 0.63 with a mean gap of
$0.14/mi — the simulator includes genuine headhaul/backhaul asymmetry. That's why
direction-aware regional features are justified.

---

## 5. Weight, trucks, and regulation

- **80,000 lb** federal gross vehicle weight limit (truck + trailer + cargo). Tractor +
  empty trailer ≈ 33–37k lb → **max payload ≈ 43–45k lb**. Our weights (up to ~47.5k before
  the sign-flip fix) sit at this boundary — plausible.
- **FTL pricing is per-truck, not per-pound.** The shipper buys the whole trailer; whether it
  carries 15k or 40k lb barely changes cost (slight fuel effect). This is why our
  `corr(weight, rate) = 0.036` is *realistic*, not a data flaw — weight prices LTL
  (less-than-truckload) freight, not truckload.
- **Hours of Service (HOS)** — drivers may drive max 11 h/day within a 14-h window → a solo
  driver covers ~500–650 mi/day. Distance therefore maps to transit days (1-day vs 2-day vs
  3-day freight), which is how shippers think about lanes.

---

## 6. Seasonality — what the calendar does to rates

This is the section that matters most for a Nov–Dec forecast:

- **Produce season (Feb–Jul)**: harvests start in Florida/Texas in late winter, move north
  through summer. Reefer demand spikes and *pulls van rates up too* (vans get tendered
  overflow, carriers reposition toward produce regions). Peak effect typically Apr–Jul.
  Our data's June peak (~+10–12% over January, all equipment types) matches this shape.
- **Retail peak / holiday season (Oct–mid Dec)**: inventory pushes for the holidays.
  **October is typically the highest-rate month of the year; spot rates often peak late
  October to mid-November**, stay elevated through the first 2–3 weeks of December (holiday
  food = reefer premium), then **fall off a cliff after Christmas** — Christmas–New Year is
  one of the deadest weeks in trucking.
- **Known calendar shocks**: DOT "International Roadcheck" inspection week (May — many
  carriers park, capacity tightens), July 4th, Thanksgiving week, and quarter-ends.
- **Day-of-week**: most freight books Mon–Thu; weekend postings are thin and mixed. Our
  ~2.4% Wed/Thu premium is a mild version of this.

**Implication for us:** the real-world prior says Nov rates ≥ Oct rates, with strength through
mid-Dec and a collapse in the last week of December. Our train data (Jan–Oct, soft June peak,
easing into Oct) gives the model no direct evidence of this pattern — this is exactly the
"no fold sees a full seasonal cycle" risk from
[Aug_22_Data_splitting.md](progress/Aug_22_Data_splitting.md). Whether the simulator encodes
a holiday ramp is unknowable from train alone; the December chart's shape deserves a sanity
check against this prior (smooth continuation vs. an implausible cliff).

---

## 7. Market indices (what `market_index` is imitating)

Real-world analogs of our mystery columns:

- **DAT RateView / Trendlines** — lane-level spot benchmarks from $150B+ of transactions.
- **FreightWaves SONAR** — high-frequency indices: OTRI (tender rejections), OTVI (volumes).
- **Cass Freight Index** — monthly shipments/expenditures.
- **Load-to-truck ratio** — per-market, per-equipment demand gauge.

Our EDA found `market_index` is load-level noise around the daily market level (its daily
mean tracks daily median $/mile at r = 0.82) — i.e., it simulates "a noisy observation of
today's market index for this load." Real pricing engines use exactly such an index as an
anchor and price loads as offsets from it — which is the two-stage model design (daily index
× per-load offset) the EDA now motivates. `quote_signal` shows no signal even conditionally;
its real-world analog is unclear — treat as a distractor.

---

## 8. Column-by-column mapping

| Column | Domain concept | Our EDA verdict |
|---|---|---|
| `posted_rate` | All-in spot offer on a load board ($) | Target; log-skewed like real rate data |
| `distance` | Routed practical miles | Priced ~linearly ($/mi flat — synthetic quirk); floored at 70 mi |
| `pickup`/`delivery` (+lat/lon) | Lane origin/destination, market geography | Coords jittered/synthetic; direction asymmetry real |
| `equipment` | Trailer type (van/reefer/flatbed) | Reefer > Flatbed > Van — matches reality |
| `weight` | Payload (lb), capped ~45k by law | Near-zero effect — realistic for FTL |
| `date` | Pickup date → seasonality, day-of-week | June peak ≈ produce season; Nov–Dec prior unobserved |
| `market_index` | Noisy per-load read of the daily market level (à la DAT/SONAR) | r=0.82 daily tracking; absent at inference |
| `quote_signal` | Unclear (quote-behavior signal?) | No conditional signal — likely distractor |

---

## 9. Glossary

- **Broker** — intermediary matching shipper loads to carrier trucks (Spotter's customers' world).
- **Carrier** — company operating the trucks; **owner-operator** — single-truck carrier.
- **Shipper** — the freight owner paying for transport.
- **Load board** — marketplace where loads/trucks are posted (DAT, Truckstop.com).
- **FTL / LTL** — full truckload (one shipper fills the trailer) vs less-than-truckload
  (shared trailer, priced per pound/class).
- **Lane** — directed origin→destination pair.
- **Linehaul** — base rate excluding fuel/accessorials.
- **FSC** — fuel surcharge.
- **RPM** — rate per mile, the universal comparison unit.
- **Deadhead** — unpaid empty miles between loads.
- **Headhaul/backhaul** — the strong/weak direction of an imbalanced lane.
- **Load-to-truck ratio** — posted loads ÷ posted trucks; local demand gauge.
- **Tender rejection** — carrier refusing a contracted load; spot-market pressure signal.
- **HOS** — Hours of Service; caps daily driving at 11 h.
- **Freight recession** — 2022–25 period of spot rates at/below operating cost.

---

## Sources

- [DAT: spot rates and linehaul/FSC breakdown (Jul–Aug 2026)](https://www.dat.com/company/news-events/news-releases/dat-contract-van-and-reefer-rates-make-record-june-to-july-gains) ·
  [DAT via GlobeNewswire: van/flatbed spot rates](https://www.globenewswire.com/news-release/2026/07/09/3324951/0/en/dat-dry-van-spot-rates-top-contract-for-first-time-since-february-2022-flatbed-rates-hit-record-high.html) ·
  [FleetOwner: YoY spot rates by equipment](https://www.fleetowner.com/news/rates/news/55369531/dry-van-reefer-and-flatbed-spot-rates-increase-year-over-year-despite-lower-loads)
- [ATRI: Operational Costs of Trucking 2025 update](https://truckingresearch.org/2026/07/new-atri-report-details-accelerating-costs-and-low-profitability-despite-cuts/) ·
  [ATRI 2024 costs](https://truckingresearch.org/2025/07/new-atri-report-shows-trucking-profitability-severly-squeezed-by-high-costs-low-rates/)
- [OTR Solutions: produce season effects](https://otrsolutions.com/blog/how-the-produce-season-affects-the-freight-industry) ·
  [C.H. Robinson: produce season guide](https://www.chrobinson.com/en-us/resources/blog/produce-season-what-shippers-should-know/) ·
  [Truck Dispatch Experts: seasonal freight calendar (Oct/Nov peak, post-Christmas dead week)](https://truckdispatchexperts.com/resources/seasonal-freight-calendar/)
- [O Trucking: how spot rates work](https://otrucking.com/resources/guides/how-spot-market-rates-work/) ·
  [Bobtail: truckload rates & length-of-haul](https://www.bobtail.com/blog/truck-load-rates/) ·
  [LearnDispatch: short vs long haul economics](https://www.learndispatch.com/long-haul-vs-short-haul-decoding-truck-dispatching/)
