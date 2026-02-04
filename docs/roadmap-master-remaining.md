# Master Roadmap: Remaining Features

*Consolidated list of unimplemented features from all roadmap files, prioritized by importance.*

## Priority Scoring System

- **P0 (Critical)**: Core functionality that blocks major use cases
- **P1 (High)**: Important features that enhance core experience
- **P2 (Medium)**: Nice-to-have features that add polish
- **P3 (Low)**: Future enhancements with minimal impact

## Data & Analytics Features

### P1: Player Stats for Props Betting
**Source**: `roadmap-data-scope.md`
**Description**: Individual player statistics for proposition betting predictions
**Impact**: Enables player props betting analysis (fantasy sports integration)
**Effort**: High (requires new data sources and scraping)
**Dependencies**: Player tracking APIs, advanced data processing

### P2: Bracket State from ESPN Data
**Source**: `roadmap-bracket-simulation.md` (Next Steps)
**Description**: Automated bracket loading from ESPN instead of manual entry
**Impact**: Eliminates manual bracket setup for tournaments
**Effort**: Medium (ESPN API integration)
**Dependencies**: ESPN API access, bracket parsing logic

## User Experience Features

### P1: ROI Tracker
**Source**: `roadmap-extended-features.md`
**Description**: Track actual betting performance and return on investment over time
**Impact**: Essential for serious bettors to measure system effectiveness
**Effort**: Medium (database integration, performance tracking)
**Dependencies**: Historical betting data, results tracking

### P1: Export Picks Functionality
**Source**: `roadmap-extended-features.md`
**Description**: Export betting recommendations to CSV/PDF for sportsbook submission
**Impact**: Critical for actual betting workflow
**Effort**: Low (data export functions)
**Dependencies**: Pick formatting, file generation

## Visualization Features

### P2: Plotly Heatmap for Probabilities
**Source**: `roadmap-bracket-visualization.md` (Next Steps)
**Description**: Interactive probability heatmaps showing advancement chances
**Impact**: Better visual understanding of tournament probabilities
**Effort**: Medium (Plotly integration, data visualization)
**Dependencies**: Simulation results, interactive charting

### P3: SVG Template for Printable Bracket
**Source**: `roadmap-bracket-visualization.md` (Next Steps)
**Description**: Vector-based bracket template for printing and sharing
**Impact**: Enhanced sharing and offline viewing
**Effort**: Low (SVG generation, template design)
**Dependencies**: Bracket data export, vector graphics

## Implementation Priority Matrix

| Feature | Business Value | Technical Effort | Timeline | Dependencies |
|---------|----------------|------------------|----------|--------------|
| ROI Tracker | High | Medium | Q1 2026 | Database, results API |
| Export Picks | High | Low | Q1 2026 | File I/O, formatting |
| Player Stats | Medium | High | Q2 2026 | New data sources |
| ESPN Bracket Loading | Medium | Medium | Q1 2026 | ESPN API |
| Probability Heatmaps | Low | Medium | Q2 2026 | Plotly expertise |
| Printable Bracket | Low | Low | Q2 2026 | SVG templates |

## Quick Wins (Low Effort, High Impact)

1. **Export Picks Functionality** - Simple data export, immediate user value
2. **Printable Bracket SVG** - Easy implementation, nice user feature
3. **ROI Tracker** - Medium effort but essential for credibility

## Major Projects (High Effort, High Value)

1. **Player Stats Integration** - Opens new betting markets, significant data engineering
2. **ESPN Bracket Automation** - Reduces manual work, improves reliability

## Dependencies & Prerequisites

### Data Sources
- ESPN API access for live bracket data
- Player tracking services (SportRadar, etc.)
- Enhanced results APIs for ROI tracking

### Technical Infrastructure
- Database for performance tracking
- File export capabilities
- Advanced visualization libraries

### External Services
- Sportsbook APIs for automated result tracking
- Print-friendly template generation

## Success Metrics

- **ROI Tracker**: Track actual betting performance vs predictions
- **Export Picks**: Successful sportsbook submissions
- **Player Stats**: Accuracy improvement on player props
- **ESPN Integration**: Reduced manual bracket setup time
- **Visualizations**: User engagement and comprehension metrics

## Risk Assessment

- **High Risk**: Player stats integration (data availability, API costs)
- **Medium Risk**: ESPN API integration (rate limits, terms of service)
- **Low Risk**: Export functionality, printable brackets, ROI tracking

---

*Generated from analysis of all roadmap files in `/docs/` folder. Last updated: February 3, 2026*</content>
<parameter name="filePath">c:\Users\gmalb\Downloads\march-madness\docs\roadmap-master-remaining.md