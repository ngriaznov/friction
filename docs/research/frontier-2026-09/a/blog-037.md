# Putting My Smart Home in Its Own Room: A VLAN Setup That Actually Stuck

For about three years, my home network was one big flat room. My work laptop, my partner's desktop, the NAS with a decade of family photos, and forty-odd "smart" devices all shared the same 192.168.1.0/24 subnet. The smart plugs, the robot vacuum, the two no-name cameras, the fridge that for some reason needs Wi-Fi. All of them could see everything else.

I knew this was a bad idea in the abstract. What finally got me to act was watching one of those cheap cameras in my router's connection log. It was chattering to an IP range I didn't recognize every few seconds, around the clock. It probably wasn't doing anything malicious. But I couldn't say for sure, and it was sitting one hop away from my backups.

So I spent a couple of weekends splitting things up. Here's what I built, what I bought, and what I'd do differently.

## The plan

I settled on four networks:

- **VLAN 10, Trusted:** laptops, desktops, phones, the NAS.
- **VLAN 20, IoT:** smart plugs, bulbs, the vacuum, the thermostat, TVs and streaming boxes.
- **VLAN 30, Cameras:** security cameras only, with no internet access at all.
- **VLAN 40, Guest:** visitors' phones, internet only.

The rules are simple. Trusted can reach anything. IoT can reach the internet and nothing else, and it can only answer connections that Trusted starts. Cameras can talk to the NVR on the NAS and nothing else. Guest is walled off completely.

## The hardware

**Router/firewall: a fanless mini PC running OPNsense.** I bought a four-port Intel N100 box, the kind sold under a dozen brand names, with 8 GB of RAM and a small SSD. It cost about $180. My ISP's combo modem-router went into bridge mode and now just passes the connection through. I went with OPNsense over a consumer router because consumer firmware tends to treat VLANs as an afterthought, and I wanted real firewall rules I could read and audit.

**Switch: TP-Link TL-SG108E.** It's an 8-port "easy smart" managed switch that costs about $30. It does 802.1Q VLAN tagging, which is all I needed. The web UI is clunky and there's no proper CLI, but after I configured it I've barely touched it. One port carries a trunk up to the router, and the other ports are either tagged trunks or untagged access ports on a single VLAN.

**Wireless: one Ubiquiti U6 Lite access point.** This was the piece that made everything work. Most IoT devices are Wi-Fi only, so I needed several SSIDs, each tied to its own VLAN. The U6 Lite handles that without trouble. I run the UniFi Network controller in a Docker container on the NAS, so I didn't have to buy a separate controller.

Total spend was around $330, not counting cables and an afternoon of swearing.

## The parts that bit me

**Discovery protocols.** The first thing that broke was casting from my phone to the TV. Chromecast and AirPlay find devices with mDNS, and mDNS doesn't cross subnets. The fix was the `os-mdns-repeater` plugin in OPNsense, set to repeat between Trusted and IoT only. Some people call that a security hole. I decided it was an acceptable one.

**Devices that insist on 2.4 GHz.** About half my smart plugs refused to join a network that broadcasts both bands under one name. I gave the IoT SSID 2.4 GHz only and the pairing problems went away.

**Firewall rule order.** OPNsense evaluates rules top to bottom, first match wins. I originally put "allow IoT to any" above "block IoT to RFC1918 networks," which quietly defeated the whole point. Now the block rules go first on every interface, and I test them from a laptop temporarily connected to each VLAN.

**Locking myself out.** At one point I changed the switch port that my management laptop was plugged into and lost access to the switch UI. I had to factory reset it. Keep one untagged port on your management VLAN, label it with tape, and don't touch it.

## Was it worth it?

Yes, and not only for security. Having IoT on its own subnet made the network easier to understand. When something is misbehaving, I can see all the IoT traffic in one place. I found out the vacuum was making DNS requests every two seconds, and a smart bulb had been failing to update its firmware for eight months.

The cameras have no internet access at all now, which I should have done from the start. They record to the NAS, and that's their entire job.

If you're thinking about trying this, start smaller than I did. A guest network and an IoT network on a decent access point already gets you most of the benefit. Add the rest once you've learned how your devices actually behave.
