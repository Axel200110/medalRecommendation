export default async (request) => {
  try {
    if (request.method !== 'POST') {
      return json({ message: 'Method not allowed' }, 405);
    }

    const apiKey = process.env.BREVO_API_KEY;
    const senderEmail = process.env.BREVO_SENDER_EMAIL;
    const senderName = process.env.BREVO_SENDER_NAME || 'TALENT.PREMIUM';

    if (!apiKey || !senderEmail) {
      console.error('Missing email environment variables.');
      return json({ message: 'Email service is not configured.' }, 500);
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return json({ message: 'Invalid JSON body.' }, 400);
    }

    const { toEmail, toName, subject, bodyContent } = payload || {};
    if (!toEmail || !subject || !bodyContent) {
      return json({ message: 'Missing required email fields.' }, 400);
    }

    const response = await fetch('https://api.brevo.com/v3/smtp/email', {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'api-key': apiKey,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        sender: { name: senderName, email: senderEmail },
        to: [{ email: toEmail, name: toName || toEmail }],
        subject,
        htmlContent: bodyContent,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      console.error('Brevo email delivery failed.', response.status, errorText);
      return json({ message: 'Email delivery failed.' }, response.status);
    }

    return json({ ok: true });
  } catch (error) {
    console.error('send-email function crashed.', error);
    return json({ message: 'Email function failed.' }, 500);
  }
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}
