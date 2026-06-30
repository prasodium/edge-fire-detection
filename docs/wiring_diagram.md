# Wiring Diagram

## Components

- Raspberry Pi 5 (8GB)
- Raspberry Pi Camera Module 3 NoIR Wide (CSI ribbon cable, CAM0 or CAM1 port)
- Active buzzer (5V, GPIO-driven via transistor or buzzer module with built-in driver)
- Relay module (5V, opto-isolated, 1-channel) driving siren / building FACP dry contact
- Status LED (with inline resistor, or an LED module with built-in resistor)
- 5V/5A (25W) official Raspberry Pi 5 USB-C power supply (15W budget leaves headroom for camera + GPIO peripherals)

## GPIO pin assignment (matches `configs/alarm.yaml`)

| Function | BCM GPIO | Physical pin | Notes |
|---|---|---|---|
| Buzzer | GPIO17 | Pin 11 | Direct-drive if buzzer module has its own transistor; otherwise use a 2N2222/NPN driver stage |
| Relay (siren / FACP interlock) | GPIO27 | Pin 13 | Opto-isolated relay module strongly recommended - isolates Pi GPIO from siren/FACP voltage domain |
| Status LED | GPIO22 | Pin 15 | Through a 330Ω resistor if not using a pre-resistored LED module |
| Ground (buzzer/relay/LED common) | GND | Pin 9, 14, 20, 25, etc. | Use any of the Pi's multiple GND pins |

## ASCII wiring diagram

```
                         Raspberry Pi 5 GPIO Header (40-pin)
                         ┌─────────────────────────────────┐
                         │  3V3  (1) (2)  5V                │
                         │  ...  (..)(..) ...               │
              Buzzer +  ─┼─ GPIO17 (11)                     │
                         │                                  │
             Relay IN   ─┼─ GPIO27 (13)                     │
                         │                                  │
             Status LED ─┼─ GPIO22 (15)──[330Ω]──▶|── GND   │
                         │                                  │
                  GND   ─┼─ GND (14/20/25/30/34/39)         │
                         └─────────────────────────────────┘
                                  │            │
                                  ▼            ▼
                         ┌────────────┐  ┌──────────────────┐
                         │  Buzzer    │  │  Relay Module     │
                         │  (5V, with │  │  (opto-isolated)  │
                         │  driver)   │  │   COM ── NO ──▶   │  to siren / FACP
                         └────────────┘  │                   │  dry-contact input
                                         └──────────────────┘

                         CSI Connector (CAM0)
                         ┌─────────────────────────────────┐
                         │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │ ◀── 22-pin CSI ribbon
                         └─────────────────────────────────┘
                                  │
                                  ▼
                  Camera Module 3 NoIR Wide (IMX708)
```

## Power budget sanity check

| Component | Typical draw | Notes |
|---|---|---|
| Raspberry Pi 5 (CPU-bound inference workload) | ~7-9.5W | Varies with clock/thermal state; this is the dominant load |
| Camera Module 3 | ~0.3-0.4W | CSI-powered from the Pi |
| Buzzer (active) | ~0.1-0.3W (only during alarm) | Negligible average draw - alarms are rare events |
| Relay module (coil energized) | ~0.2-0.4W (only during alarm) | |
| Status LED | <0.05W | |
| **Total continuous (no alarm)** | **~7.5-10W** | Within the 15W budget with margin |
| **Total during alarm burst** | **~8-11W** | Still within budget |

Use the official 27W USB-C supply (or better) even though typical draw is
lower — transient inrush during boot and SSD/USB peripheral attachment can
exceed steady-state draw, and undervoltage is a common cause of camera/CSI
instability and silent throttling on the Pi 5.

## Safety notes

- **Always opto-isolate the relay** if it interfaces with mains-voltage
  siren circuitry or an existing fire alarm control panel (FACP) — never
  wire GPIO directly to anything above 5V logic level.
- **This system is a supplementary early-detection aid, not a certified
  fire alarm system.** If integrating with a building's FACP, the relay
  output should be treated as an auxiliary input/trouble signal per local
  fire code (e.g. NFPA 72 in the US), reviewed and signed off by a licensed
  fire protection engineer before connecting to life-safety circuits. See
  `docs/deployment_guide.md` "Regulatory & Compliance" section.
