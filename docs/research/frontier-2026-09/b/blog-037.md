# Putting the Smart Home in Its Own Corner: VLANs for IoT Isolation

I resisted this project for about two years. Every time I added another device to the house, a smart plug here, a robot vacuum there, a doorbell camera, a thermostat, I had the same uneasy thought: all of these things are sitting on the same network as my laptop, my NAS, and the machine I do my taxes on. A cheap Wi-Fi bulb with firmware that hasn't been updated since 2021 has a direct line to everything I care about.

This spring I finally fixed it. Here's what I built, what I bought, and what I'd do differently.

## The goal

Three separate networks, on one set of physical hardware:

1. **Trusted**: my computers, phones, the NAS, and the printer.
2. **IoT**: bulbs, plugs, cameras, the thermostat, the TV, the vacuum.
3. **Guest**: friends' devices, isolated from everything.

The IoT network should be able to reach the internet but not initiate connections to Trusted. Trusted should be able to reach IoT so I can still open the camera app or cast to the TV. Guest gets internet and nothing else.

## Hardware

I wanted gear that supported 802.1Q VLAN tagging end to end without paying enterprise prices.

**Router: a small fanless mini PC running OPNsense.** A four-port Intel N100 box, the kind that shows up on every "cheap firewall" list. Around $200. I considered a consumer router with VLAN support, but I wanted real firewall rules, not a checkbox that says "isolate guests."

**Switch: a TP-Link TL-SG108E.** An eight-port managed switch for about $30. It does VLAN tagging, it has a web interface that looks like it was designed in 2008, and it has been completely reliable. If you have more than eight wired devices you'll want the 16-port version, but for me eight was enough.

**Access point: a Ubiquiti U6 Lite.** This was the piece I was most worried about. You need an AP that can broadcast multiple SSIDs and tag each one to a different VLAN. The U6 Lite does exactly that, and it's powered over Ethernet so there's one cable to it. I run the UniFi controller in a Docker container on the NAS rather than buying a Cloud Key.

Total spend was a little under $350, most of which was the router.

## Setting it up

The order that worked for me:

**1. Define the VLANs on the router.** In OPNsense I created VLAN 10 (Trusted), VLAN 20 (IoT), and VLAN 30 (Guest), all on the LAN interface. Each gets its own subnet and DHCP range: 10.0.10.0/24, 10.0.20.0/24, 10.0.30.0/24. I left the untagged LAN as a management network I use only from a wired connection.

**2. Configure the switch.** The port that connects to the router and the port that connects to the AP are both set as trunk ports, tagged for all three VLANs. Every other port is an access port assigned to a single VLAN. The NAS and my desktop are on VLAN 10 ports; the wired TV is on VLAN 20. This is where I made my first mistake: I forgot to set the PVID on the access ports, so untagged traffic from the desktop was landing on the wrong network. The switch's UI treats VLAN membership and PVID as two separate screens, and you need both.

**3. Set up the SSIDs.** Three wireless networks in the UniFi controller, each tagged with its VLAN. The IoT SSID is 2.4 GHz only, because half of my smart plugs can't see 5 GHz and it's simpler to not give them the option.

**4. Write the firewall rules.** This is the part that actually matters. By default OPNsense blocks traffic between interfaces, so the starting point is "nothing talks to anything." Then:

- Trusted: allow all outbound.
- IoT: allow to internet, block to Trusted and Guest. One exception: allow established connections back, so when my phone opens a camera stream the reply packets get through.
- Guest: allow to internet only, block all RFC1918 ranges.

**5. Deal with mDNS.** Casting and AirPlay discovery rely on multicast, which doesn't cross VLANs. I installed the mDNS repeater plugin in OPNsense and let it reflect between VLAN 10 and 20. Without this, the TV doesn't show up in the cast menu. With it, everything works exactly as before.

## What broke

The doorbell camera lost its ability to send clips to the NAS, because that's an IoT-to-Trusted connection. I added a single narrow rule allowing that camera's IP to reach the NAS on one port. The vacuum's app took two days to work reliably, until I realised it was doing local discovery via UDP broadcast and needed the same mDNS treatment.

## Was it worth it?

Yes. I can now add any gadget to the house without thinking about it. If a device turns out to be phoning home to somewhere unpleasant, the worst it can reach is a network full of other light bulbs. And the setup, once done, has needed zero maintenance in four months.

If you only do one thing: buy a managed switch and an AP that supports multiple SSIDs. Everything else is configuration.
