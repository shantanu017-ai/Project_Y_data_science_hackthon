# LinkedIn post — VoltRelay diagnostic

**Status:** ready to publish once the site URL is inserted.
**Length:** ~1,250 characters, under LinkedIn's 3,000-character limit.
**Data note:** the dataset is synthetic and provided by the hackathon. Keep the disclaimer.

---

A battery-swapping network can grow 3× and get worse at the same time. That is what the data said.

I analysed 3.8M swap attempts across six Indian cities for the Gradient Learnings Data Analytics Hackathon (synthetic VoltRelay dataset, Jan 2024 – Jun 2025).

Completed swaps grew 2.9×. Failed attempts grew 6.0×. The surprise was not the growth. It was where the failures came from:

→ Heat. 17.3% of all failures are attributable to temperatures above 35°C. At 45°C+, first-generation cabinets take 156 minutes to recharge and fail at 15.3%, versus 44 minutes and 5.1% for the newer generation. 121 of 138 stations performed worse on hot days.

→ One battery supplier. Kyron's 2W packs wore out at ₹30.79 per swap against ₹9.56 for peers. All 1,385 packs are retired. Avoidable wear: about ₹12.7M.

→ One partner. ZipDrop contributes 549,390 swaps at ₹6.17 per swap. The network average is ₹16.53.

→ The retention cost. Among riders still active in week four, D60 retention falls from 96.9% with no early failures to 90.4% above 20% exposure. Riders forgive a bad day, not a bad month.

What I would do first: cool the Gen1 fleet, quarantine three battery lots, and reprice the largest partner. Not another station wave.

The project: a Colab-ready notebook, a full report, and an editorial findings site. Built with DuckDB, Python and a lot of stratified comparisons.

Site: https://voltrelay-diagnostic-4xc7jjkf85z.qoder.website

Dataset is synthetic and provided by the hackathon. Figures are illustrative, not a real company.

#DataAnalytics #Hackathon #BatterySwapping #EV #DuckDB #Python #ProductAnalytics #India

---

## Posting checklist

- [ ] Open the site link while signed in to Qoder, then share it from Sites Settings if a wider audience is needed (the site is currently private).
- [ ] Attach the growth chart (`01_kpi_growth.png`) and the heat chart (`02_heat_curve.png`) as a 2-image carousel.
- [ ] Tag the hackathon organiser if they have a LinkedIn page.
- [ ] Put the notebook and report links in the first comment to keep the post clean.
- [ ] Do not change the synthetic-data disclaimer.
