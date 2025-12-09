from obspy import Catalog


def get_catalog_picks(catalog, client):

    new_catalog = Catalog()

    failed_requests = []

    for event in catalog:

        try:
            # Get event using eventid
            eventid = event.resource_id.split("/")[-1]
            events = client.get_events(eventid, includearrivals=True)
            new_catalog += events
            print(f"  Retrieved event {eventid}")

        except Exception as e:
            print(f"  Failed to retrieve event {event.resource_id}: {e}")
            failed_requests.append(event.resource_id)
            continue

    print(f"Successfully retrieved {len(catalog)} events")
    if failed_requests:
        print(f"Failed to retrieve {len(failed_requests)} events: {failed_requests}")

    return new_catalog